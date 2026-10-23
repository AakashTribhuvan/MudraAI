# Brings the Chrome window of the given process to the front (Alt-tap workaround for the foreground lock) and sends keys.
param([int]$ProcId, [string]$Keys)
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class W {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
  [DllImport("user32.dll")] public static extern void keybd_event(byte k, byte s, uint f, UIntPtr e);
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
}
"@
$h = (Get-Process -Id $ProcId).MainWindowHandle
if ($h -eq [IntPtr]::Zero) { "no window"; exit 1 }
[W]::keybd_event(0x12, 0, 0, [UIntPtr]::Zero); [W]::keybd_event(0x12, 0, 2, [UIntPtr]::Zero)
[W]::ShowWindow($h, 9) | Out-Null
$ok = [W]::SetForegroundWindow($h)
Start-Sleep -Milliseconds 600
$front = ([W]::GetForegroundWindow() -eq $h)
if ($front) { (New-Object -ComObject WScript.Shell).SendKeys($Keys) }
"foreground=$front"
