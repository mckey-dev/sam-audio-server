"""Windows の専用 GPU メモリ（タスクマネージャに近い値）を DXGI / PDH から読む。"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from ctypes import wintypes

_DXGI_ERROR_NOT_FOUND = 0x887A0002
_DXGI_MEMORY_SEGMENT_GROUP_LOCAL = 0
_PDH_FMT_DOUBLE = 0x00000200
_PDH_MORE_DATA = 0x800007D2

_lock = threading.Lock()
_pdh_query = ctypes.c_void_p()
_pdh_counter = ctypes.c_void_p()
_pdh_ready = False


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _LUID(ctypes.Structure):
    _fields_ = [
        ("LowPart", wintypes.DWORD),
        ("HighPart", wintypes.LONG),
    ]


class _DXGI_ADAPTER_DESC(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
    ]


class _DXGI_QUERY_VIDEO_MEMORY_INFO(ctypes.Structure):
    _fields_ = [
        ("Budget", ctypes.c_uint64),
        ("CurrentUsage", ctypes.c_uint64),
        ("AvailableForReservation", ctypes.c_uint64),
        ("CurrentReservation", ctypes.c_uint64),
    ]


class _PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _fields_ = [
        ("CStatus", wintypes.DWORD),
        ("doubleValue", ctypes.c_double),
    ]


class _PDH_FMT_COUNTERVALUE_ITEM(ctypes.Structure):
    _fields_ = [
        ("szName", ctypes.c_wchar_p),
        ("FmtValue", _PDH_FMT_COUNTERVALUE),
    ]


def _make_guid(text: str) -> _GUID:
    raw = bytes.fromhex(text.strip("{}").replace("-", ""))
    guid = _GUID()
    guid.Data1 = int.from_bytes(raw[0:4], "big")
    guid.Data2 = int.from_bytes(raw[4:6], "big")
    guid.Data3 = int.from_bytes(raw[6:8], "big")
    guid.Data4[:] = raw[8:16]
    return guid


_IID_FACTORY1 = _make_guid("770AAE78-F26F-4DBA-A829-253C83D1B387")
_IID_ADAPTER3 = _make_guid("645967A4-1392-4310-A798-8053CE3E93FD")


def query_os_vram(device_name: str | None = None) -> tuple[float, float] | None:
    """このプロセスの専用 GPU 使用量とアダプタ専用容量を GiB で返す。失敗時は None。"""
    if sys.platform != "win32":
        return None
    try:
        adapter = _find_adapter(device_name)
        if adapter is None:
            return None
        used_bytes, total_bytes, luid_high, luid_low, dxgi_usage = adapter
        pdh_used = _pdh_process_dedicated(luid_high, luid_low)
        if pdh_used is not None:
            used_bytes = pdh_used
        elif dxgi_usage > 0:
            used_bytes = dxgi_usage
        if total_bytes <= 0:
            return None
        return used_bytes / (1024**3), total_bytes / (1024**3)
    except Exception:
        return None


def _find_adapter(
    device_name: str | None,
) -> tuple[int, int, int, int, int] | None:
    """一致する DXGI アダプタの (ダミー使用量, 専用容量, LUID, DXGI CurrentUsage) を返す。"""
    ole32 = ctypes.windll.ole32
    ole32.CoInitializeEx(None, 0x0)

    factory = ctypes.c_void_p()
    create = ctypes.windll.dxgi.CreateDXGIFactory1
    create.argtypes = [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_long
    if create(ctypes.byref(_IID_FACTORY1), ctypes.byref(factory)) != 0 or not factory.value:
        return None

    want = (device_name or "").strip()
    matched: tuple[int, int, int, int, int] | None = None
    fallback: tuple[int, int, int, int, int] | None = None
    index = 0
    while True:
        adapter = ctypes.c_void_p()
        hr = _call(factory, 7, ctypes.c_uint, index, ctypes.c_void_p, ctypes.byref(adapter))
        if (hr & 0xFFFFFFFF) == _DXGI_ERROR_NOT_FOUND or hr != 0:
            break
        desc = _DXGI_ADAPTER_DESC()
        _call(adapter, 8, ctypes.c_void_p, ctypes.byref(desc))
        name = desc.Description
        dedicated = int(desc.DedicatedVideoMemory)
        if "basic render" in name.lower() or dedicated <= 0:
            _call(adapter, 2)
            index += 1
            continue
        usage = _query_usage(adapter)
        info = (0, dedicated, int(desc.AdapterLuid.HighPart), int(desc.AdapterLuid.LowPart), usage)
        if _names_match(want, name):
            matched = info
            _call(adapter, 2)
            break
        if fallback is None or dedicated > fallback[1]:
            fallback = info
        _call(adapter, 2)
        index += 1
    _call(factory, 2)
    return matched or fallback


def _query_usage(adapter: ctypes.c_void_p) -> int:
    """DXGI がこのプロセスに計上している LOCAL 使用量（ドライバ込み）。"""
    adapter3 = ctypes.c_void_p()
    hr = _call(
        adapter,
        0,
        ctypes.c_void_p,
        ctypes.byref(_IID_ADAPTER3),
        ctypes.c_void_p,
        ctypes.byref(adapter3),
    )
    if hr != 0 or not adapter3.value:
        return 0
    info = _DXGI_QUERY_VIDEO_MEMORY_INFO()
    hr = _call(
        adapter3,
        14,
        ctypes.c_uint,
        0,
        ctypes.c_int,
        _DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
        ctypes.c_void_p,
        ctypes.byref(info),
    )
    _call(adapter3, 2)
    if hr != 0:
        return 0
    return int(info.CurrentUsage)


def _pdh_process_dedicated(luid_high: int, luid_low: int) -> int | None:
    """タスクマネージャの「専用 GPU メモリ」と同じ PDH カウンタを読む。"""
    with _lock:
        if not _ensure_pdh():
            return None
        pdh = ctypes.windll.pdh
        pdh.PdhCollectQueryData(_pdh_query)
        size = wintypes.DWORD(0)
        count = wintypes.DWORD(0)
        hr = pdh.PdhGetFormattedCounterArrayW(
            _pdh_counter,
            _PDH_FMT_DOUBLE,
            ctypes.byref(size),
            ctypes.byref(count),
            None,
        )
        if (hr & 0xFFFFFFFF) not in (0, _PDH_MORE_DATA) or size.value == 0:
            return None
        buf = (ctypes.c_byte * size.value)()
        hr = pdh.PdhGetFormattedCounterArrayW(
            _pdh_counter,
            _PDH_FMT_DOUBLE,
            ctypes.byref(size),
            ctypes.byref(count),
            buf,
        )
        if hr != 0:
            return None
        needle = f"pid_{os.getpid()}_luid_0x{luid_high & 0xFFFFFFFF:08X}_0x{luid_low & 0xFFFFFFFF:08X}_"
        items = ctypes.cast(buf, ctypes.POINTER(_PDH_FMT_COUNTERVALUE_ITEM))
        total = 0.0
        found = False
        for i in range(count.value):
            name = items[i].szName or ""
            if needle.lower() in name.lower():
                total += float(items[i].FmtValue.doubleValue)
                found = True
        if not found:
            return None
        return int(total)


def _ensure_pdh() -> bool:
    """PDH クエリを一度だけ開き、専用メモリカウンタを載せる。"""
    global _pdh_ready
    if _pdh_ready:
        return True
    pdh = ctypes.windll.pdh
    pdh.PdhOpenQueryW.argtypes = [
        wintypes.LPCWSTR,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    pdh.PdhOpenQueryW.restype = ctypes.c_uint32
    pdh.PdhAddEnglishCounterW.argtypes = [
        ctypes.c_void_p,
        wintypes.LPCWSTR,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    pdh.PdhAddEnglishCounterW.restype = ctypes.c_uint32
    pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
    pdh.PdhCollectQueryData.restype = ctypes.c_uint32
    pdh.PdhGetFormattedCounterArrayW.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    pdh.PdhGetFormattedCounterArrayW.restype = ctypes.c_uint32
    if pdh.PdhOpenQueryW(None, None, ctypes.byref(_pdh_query)) != 0:
        return False
    path = ctypes.c_wchar_p(r"\GPU Process Memory(*)\Dedicated Usage")
    if pdh.PdhAddEnglishCounterW(_pdh_query, path, None, ctypes.byref(_pdh_counter)) != 0:
        return False
    pdh.PdhCollectQueryData(_pdh_query)
    _pdh_ready = True
    return True


def _names_match(want: str, have: str) -> bool:
    """torch のデバイス名と DXGI のアダプタ名を突き合わせる。"""
    a = want.lower().strip()
    b = have.lower().strip()
    if not a:
        return False
    if a in b or b in a:
        return True
    for prefix in ("nvidia ", "amd ", "intel "):
        a = a.removeprefix(prefix)
        b = b.removeprefix(prefix)
    return bool(a) and (a in b or b in a)


def _call(obj: ctypes.c_void_p, slot: int, *typed_args) -> int:
    """COM vtable[slot](this, args...)。restype は HRESULT 相当の c_long。"""
    this = obj.value
    vtbl = ctypes.cast(this, ctypes.POINTER(ctypes.c_void_p))[0]
    fn = ctypes.cast(
        vtbl + slot * ctypes.sizeof(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    )[0]
    types: list = [ctypes.c_void_p]
    values: list = [this]
    for i in range(0, len(typed_args), 2):
        types.append(typed_args[i])
        values.append(typed_args[i + 1])
    proto = ctypes.WINFUNCTYPE(ctypes.c_long, *types)
    return int(proto(fn)(*values))
