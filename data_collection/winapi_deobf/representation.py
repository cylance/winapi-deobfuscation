import re
import esil


EXPR = 'expr'
REG = 'reg'


EXPR_OPERATORS = [
    ('\+', 'A'),
    ('\-', 'S'),
    ('\*', 'M'),
    ('\^', 'X'),
    ('Concat|Extract', 'B')
]

NAMES = [
    u'PostMessage', u'DeviceIoControl', u'SendDlgItemMessage', u'RegQueryValueEx', u'HeapAlloc', u'SetWindowPos',
    u'MessageBox', u'GetLocaleInfo', u'FormatMessage', u'MultiByteToWideChar', u'SetDlgItemText', u'CheckDlgButton',
    u'CreateWindowEx', u'RegCreateKeyEx', u'WideCharToMultiByte', u'CoCreateInstance', u'ReadFile', u'WriteFile',
    u'CompareString', u'SetTimer', u'LoadString', u'RegSetValueEx', u'CreateFile', u'RegOpenKeyEx', u'SendMessage'
]



def repr_int(arg):
    if arg == 0xffffffff or arg == 0x0:
        return arg

    # val = int(arg, 16)
    return str(len(arg[2:]))


def _repr(arg, idx, name, n_args=None):
    if arg.startswith('0x'):
        if n_args and idx < n_args and API_LOOKUP.get(name)[idx] == TYPE_CONST:
            return arg
        else:
            return repr_int(arg)


    if arg.startswith('unk'):
        return 'unk'

    if arg.startswith('mem'):
        return 'mem'

    if re.match(r'^arg|^var', arg):
        return 'var'

    if arg in esil.REG_MAP:
        return REG

    arg = re.sub(r'\n', '', arg)
    arg = re.sub(r'\s+', ' ', arg)

    expr = ''
    ops_found = False
    for op, k in EXPR_OPERATORS:
        # print op, arg
        if re.search(op, arg):
            expr += k
            ops_found = True
            # print '<'
            break
        else:
            expr += '_'

    if ops_found:
        return EXPR

    # if re.search(r'Concat|Extract|[\+\-\*]', arg):
    #     return EXPR

    return arg

TYPE_CONST = 'const'
TYPE_RESERVED = 'reserved'
TYPE_PTR = 'ptr'
TYPE_INT = 'int'

API_LOOKUP = {
# int CompareString(
#   _In_ LCID    Locale,
#   _In_ DWORD   dwCmpFlags,
#   _In_ LPCTSTR lpString1,
#   _In_ int     cchCount1,
#   _In_ LPCTSTR lpString2,
#   _In_ int     cchCount2
# );
    u'CompareString': [TYPE_CONST, TYPE_CONST, TYPE_PTR, TYPE_INT, TYPE_PTR, TYPE_INT],

# DWORD WINAPI FormatMessage(
#   _In_     DWORD   dwFlags,
#   _In_opt_ LPCVOID lpSource,
#   _In_     DWORD   dwMessageId,
#   _In_     DWORD   dwLanguageId,
#   _Out_    LPTSTR  lpBuffer,
#   _In_     DWORD   nSize,
#   _In_opt_ va_list *Arguments
# );
    u'FormatMessage': [TYPE_CONST, TYPE_PTR, TYPE_INT, TYPE_CONST, TYPE_PTR, TYPE_INT, TYPE_PTR],

# LONG WINAPI RegCreateKeyEx(
#   _In_       HKEY                  hKey,
#   _In_       LPCTSTR               lpSubKey,
#   _Reserved_ DWORD                 Reserved,
#   _In_opt_   LPTSTR                lpClass,
#   _In_       DWORD                 dwOptions,
#   _In_       REGSAM                samDesired,
#   _In_opt_   LPSECURITY_ATTRIBUTES lpSecurityAttributes,
#   _Out_      PHKEY                 phkResult,
#   _Out_opt_  LPDWORD               lpdwDisposition
# );
    # Note that first arg might not be a constant, but we just presume so (for now at least)
    u'RegCreateKeyEx': [TYPE_CONST, TYPE_PTR, TYPE_RESERVED, TYPE_PTR, TYPE_CONST, TYPE_CONST, TYPE_PTR, TYPE_PTR, TYPE_CONST],

# int MultiByteToWideChar(
#   _In_      UINT   CodePage,
#   _In_      DWORD  dwFlags,
#   _In_      LPCSTR lpMultiByteStr,
#   _In_      int    cbMultiByte,
#   _Out_opt_ LPWSTR lpWideCharStr,
#   _In_      int    cchWideChar
# );
    u'MultiByteToWideChar': [TYPE_CONST, TYPE_CONST, TYPE_PTR, TYPE_INT, TYPE_PTR, TYPE_INT],

# BOOL CheckDlgButton(
#   _In_ HWND hDlg,
#   _In_ int  nIDButton,
#   _In_ UINT uCheck
# );
    u'CheckDlgButton': [TYPE_INT, TYPE_INT, TYPE_CONST],

# HWND WINAPI CreateWindowEx(
#   _In_     DWORD     dwExStyle,
#   _In_opt_ LPCTSTR   lpClassName,
#   _In_opt_ LPCTSTR   lpWindowName,
#   _In_     DWORD     dwStyle,
#   _In_     int       x,
#   _In_     int       y,
#   _In_     int       nWidth,
#   _In_     int       nHeight,
#   _In_opt_ HWND      hWndParent,
#   _In_opt_ HMENU     hMenu,
#   _In_opt_ HINSTANCE hInstance,
#   _In_opt_ LPVOID    lpParam
# );
    u'CreateWindowEx': [TYPE_CONST, TYPE_PTR, TYPE_PTR, TYPE_PTR, TYPE_CONST, TYPE_INT, TYPE_INT, TYPE_INT,
                        TYPE_INT, TYPE_INT, TYPE_INT, TYPE_PTR],

# LRESULT WINAPI SendDlgItemMessage(
#   _In_ HWND   hDlg,
#   _In_ int    nIDDlgItem,
#   _In_ UINT   Msg,
#   _In_ WPARAM wParam,
#   _In_ LPARAM lParam
# );
    u'SendDlgItemMessage': [TYPE_INT, TYPE_INT, TYPE_CONST, TYPE_PTR, TYPE_PTR],

# LONG WINAPI RegQueryValueEx(
#   _In_        HKEY    hKey,
#   _In_opt_    LPCTSTR lpValueName,
#   _Reserved_  LPDWORD lpReserved,
#   _Out_opt_   LPDWORD lpType,
#   _Out_opt_   LPBYTE  lpData,
#   _Inout_opt_ LPDWORD lpcbData
# );
    u'RegQueryValueEx': [TYPE_CONST, TYPE_PTR, TYPE_RESERVED, TYPE_PTR, TYPE_PTR, TYPE_PTR],

# HANDLE WINAPI CreateFile(
#   _In_     LPCTSTR               lpFileName,
#   _In_     DWORD                 dwDesiredAccess,
#   _In_     DWORD                 dwShareMode,
#   _In_opt_ LPSECURITY_ATTRIBUTES lpSecurityAttributes,
#   _In_     DWORD                 dwCreationDisposition,
#   _In_     DWORD                 dwFlagsAndAttributes,
#   _In_opt_ HANDLE                hTemplateFile
# );

    u'CreateFile': [TYPE_PTR, TYPE_CONST, TYPE_CONST, TYPE_PTR, TYPE_CONST, TYPE_CONST, TYPE_INT],

# BOOL WINAPI SetWindowPos(
#   _In_     HWND hWnd,
#   _In_opt_ HWND hWndInsertAfter,
#   _In_     int  X,
#   _In_     int  Y,
#   _In_     int  cx,
#   _In_     int  cy,
#   _In_     UINT uFlags
# );
    u'SetWindowPos': [TYPE_INT, TYPE_CONST, TYPE_INT, TYPE_INT, TYPE_INT, TYPE_INT, TYPE_CONST],

# BOOL WINAPI ReadFile(
#   _In_        HANDLE       hFile,
#   _Out_       LPVOID       lpBuffer,
#   _In_        DWORD        nNumberOfBytesToRead,
#   _Out_opt_   LPDWORD      lpNumberOfBytesRead,
#   _Inout_opt_ LPOVERLAPPED lpOverlapped
# );
    u'ReadFile': [TYPE_INT, TYPE_PTR, TYPE_INT, TYPE_PTR, TYPE_PTR],


# LONG WINAPI RegOpenKeyEx(
#   _In_     HKEY    hKey,
#   _In_opt_ LPCTSTR lpSubKey,
#   _In_     DWORD   ulOptions,
#   _In_     REGSAM  samDesired,
#   _Out_    PHKEY   phkResult
# );
    u'RegOpenKeyEx': [TYPE_CONST, TYPE_PTR, TYPE_CONST, TYPE_CONST, TYPE_PTR],

# int WINAPI MessageBox(
#   _In_opt_ HWND    hWnd,
#   _In_opt_ LPCTSTR lpText,
#   _In_opt_ LPCTSTR lpCaption,
#   _In_     UINT    uType
# );
    u'MessageBox': [TYPE_INT, TYPE_PTR, TYPE_PTR, TYPE_CONST],

# BOOL WINAPI PostMessage(
#   _In_opt_ HWND   hWnd,
#   _In_     UINT   Msg,
#   _In_     WPARAM wParam,
#   _In_     LPARAM lParam
# );
    u'PostMessage': [TYPE_INT, TYPE_INT, TYPE_PTR, TYPE_PTR],

# int WideCharToMultiByte(
#   _In_      UINT    CodePage,
#   _In_      DWORD   dwFlags,
#   _In_      LPCWSTR lpWideCharStr,
#   _In_      int     cchWideChar,
#   _Out_opt_ LPSTR   lpMultiByteStr,
#   _In_      int     cbMultiByte,
#   _In_opt_  LPCSTR  lpDefaultChar,
#   _Out_opt_ LPBOOL  lpUsedDefaultChar
# );
    u'WideCharToMultiByte': [TYPE_CONST, TYPE_CONST, TYPE_PTR, TYPE_INT, TYPE_PTR, TYPE_INT, TYPE_PTR, TYPE_PTR],


# UINT_PTR WINAPI SetTimer(
#   _In_opt_ HWND      hWnd,
#   _In_     UINT_PTR  nIDEvent,
#   _In_     UINT      uElapse,
#   _In_opt_ TIMERPROC lpTimerFunc
# );
    u'SetTimer': [TYPE_INT, TYPE_PTR, TYPE_INT, TYPE_PTR],
#
# int WINAPI LoadString(
#   _In_opt_ HINSTANCE hInstance,
#   _In_     UINT      uID,
#   _Out_    LPTSTR    lpBuffer,
#   _In_     int       nBufferMax
# );
    u'LoadString': [TYPE_INT, TYPE_INT, TYPE_PTR, TYPE_INT],


# HRESULT CoCreateInstance(
#   _In_  REFCLSID  rclsid,
#   _In_  LPUNKNOWN pUnkOuter,
#   _In_  DWORD     dwClsContext,
#   _In_  REFIID    riid,
#   _Out_ LPVOID    *ppv
# );
    u'CoCreateInstance': [TYPE_PTR, TYPE_PTR, TYPE_CONST, TYPE_PTR, TYPE_PTR],

# BOOL WINAPI WriteFile(
#   _In_        HANDLE       hFile,
#   _In_        LPCVOID      lpBuffer,
#   _In_        DWORD        nNumberOfBytesToWrite,
#   _Out_opt_   LPDWORD      lpNumberOfBytesWritten,
#   _Inout_opt_ LPOVERLAPPED lpOverlapped
# );
    u'WriteFile': [TYPE_INT, TYPE_PTR, TYPE_INT, TYPE_PTR, TYPE_PTR],

# BOOL WINAPI DeviceIoControl(
#   _In_        HANDLE       hDevice,
#   _In_        DWORD        dwIoControlCode,
#   _In_opt_    LPVOID       lpInBuffer,
#   _In_        DWORD        nInBufferSize,
#   _Out_opt_   LPVOID       lpOutBuffer,
#   _In_        DWORD        nOutBufferSize,
#   _Out_opt_   LPDWORD      lpBytesReturned,
#   _Inout_opt_ LPOVERLAPPED lpOverlapped
# );
    u'DeviceIoControl': [TYPE_INT, TYPE_CONST, TYPE_PTR, TYPE_INT, TYPE_PTR, TYPE_INT, TYPE_PTR, TYPE_INT],

# LPVOID WINAPI HeapAlloc(
#   _In_ HANDLE hHeap,
#   _In_ DWORD  dwFlags,
#   _In_ SIZE_T dwBytes
# );
    u'HeapAlloc': [TYPE_INT, TYPE_CONST, TYPE_INT],

# int GetLocaleInfo(
#   _In_      LCID   Locale,
#   _In_      LCTYPE LCType,
#   _Out_opt_ LPTSTR lpLCData,
#   _In_      int    cchData
# );
    u'GetLocaleInfo': [TYPE_CONST, TYPE_CONST, TYPE_PTR, TYPE_INT],

# BOOL WINAPI SetDlgItemText(
#   _In_ HWND    hDlg,
#   _In_ int     nIDDlgItem,
#   _In_ LPCTSTR lpString
# );
    u'SetDlgItemText': [TYPE_INT, TYPE_INT, TYPE_PTR],

# LRESULT WINAPI SendMessage(
#   _In_ HWND   hWnd,
#   _In_ UINT   Msg,
#   _In_ WPARAM wParam,
#   _In_ LPARAM lParam
# );
    u'SendMessage': [TYPE_INT, TYPE_INT, TYPE_PTR, TYPE_PTR],

# LONG WINAPI RegSetValueEx(
#   _In_             HKEY    hKey,
#   _In_opt_         LPCTSTR lpValueName,
#   _Reserved_       DWORD   Reserved,
#   _In_             DWORD   dwType,
#   _In_       const BYTE    *lpData,
#   _In_             DWORD   cbData
# );
    u'RegSetValueEx': [TYPE_CONST, TYPE_PTR, TYPE_RESERVED, TYPE_CONST, TYPE_PTR, TYPE_INT],


}
