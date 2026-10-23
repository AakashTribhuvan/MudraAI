# Test audio with known word timings: a Windows voice speaks the text into a WAV (preceded by 1.5 s of
# silence via SSML break), and each word's audio position is written to a JSON file.
param([string]$Text, [string]$Wav, [string]$Json, [int]$Rate = 0, [string]$Voice = "")
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = $Rate
if ($Voice) { $s.SelectVoice($Voice) }
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$words = New-Object System.Collections.ArrayList
$s.add_SpeakProgress({ param($sender, $e) [void]$words.Add(@{ text = $e.Text; start_ms = $e.AudioPosition.TotalMilliseconds; chars = $e.CharacterCount }) })
$s.SetOutputToWaveFile($Wav, $fmt)
$escaped = [System.Security.SecurityElement]::Escape($Text)
$ssml = "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'><break time='1500ms'/>$escaped<break time='2500ms'/></speak>"
$s.SpeakSsml($ssml)
$s.SetOutputToNull()
$words | ConvertTo-Json | Set-Content -Path $Json -Encoding utf8
"words: " + $words.Count
