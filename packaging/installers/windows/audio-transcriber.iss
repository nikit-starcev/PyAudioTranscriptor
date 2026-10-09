#ifndef BundleDir
  #define BundleDir "..\..\..\dist\audio-transcriber"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\..\dist-installers"
#endif
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef IconFile
  #define IconFile ""
#endif

#define AppName "AudioTranscriptor"
#define AppExe "audio-transcriber.exe"

[Setup]
AppId={{7F3C2A91-4D6E-4B0A-9C7E-2A5B8D1F4E30}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=PyAudioTranscriptor
DefaultDirName={autopf}\AudioTranscriptor
DefaultGroupName=AudioTranscriptor
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename=audio-transcriber-{#AppVersion}-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64
UninstallDisplayIcon={app}\{#AppExe}
#if FileExists(IconFile)
SetupIconFile={#IconFile}
#endif

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; Flags: unchecked
Name: "assoc_audio"; Description: "Связать аудиофайлы (добавить в «Открыть с помощью»)"; Flags: unchecked
Name: "assoc_video"; Description: "Связать видеофайлы (добавить в «Открыть с помощью»)"; Flags: unchecked

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"; Parameters: "web"; IconFilename: "{app}\{#AppExe}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Parameters: "web"; IconFilename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Parameters: "web"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Registry]
Root: HKCR; Subkey: "AudioTranscriptor.audio"; ValueType: string; ValueData: "Аудиофайл (AudioTranscriptor)"; Flags: uninsdeletekey; Tasks: assoc_audio
Root: HKCR; Subkey: "AudioTranscriptor.audio\DefaultIcon"; ValueType: string; ValueData: "{app}\{#AppExe},0"; Tasks: assoc_audio
Root: HKCR; Subkey: "AudioTranscriptor.audio\shell\open\command"; ValueType: string; ValueData: """{app}\{#AppExe}"" transcribe ""%1"""; Tasks: assoc_audio
Root: HKCR; Subkey: "AudioTranscriptor.video"; ValueType: string; ValueData: "Видеофайл (AudioTranscriptor)"; Flags: uninsdeletekey; Tasks: assoc_video
Root: HKCR; Subkey: "AudioTranscriptor.video\DefaultIcon"; ValueType: string; ValueData: "{app}\{#AppExe},0"; Tasks: assoc_video
Root: HKCR; Subkey: "AudioTranscriptor.video\shell\open\command"; ValueType: string; ValueData: """{app}\{#AppExe}"" transcribe ""%1"""; Tasks: assoc_video
Root: HKCR; Subkey: ".mp3\OpenWithProgids"; ValueType: string; ValueName: "AudioTranscriptor.audio"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc_audio
Root: HKCR; Subkey: ".wav\OpenWithProgids"; ValueType: string; ValueName: "AudioTranscriptor.audio"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc_audio
Root: HKCR; Subkey: ".m4a\OpenWithProgids"; ValueType: string; ValueName: "AudioTranscriptor.audio"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc_audio
Root: HKCR; Subkey: ".flac\OpenWithProgids"; ValueType: string; ValueName: "AudioTranscriptor.audio"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc_audio
Root: HKCR; Subkey: ".ogg\OpenWithProgids"; ValueType: string; ValueName: "AudioTranscriptor.audio"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc_audio
Root: HKCR; Subkey: ".opus\OpenWithProgids"; ValueType: string; ValueName: "AudioTranscriptor.audio"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc_audio
Root: HKCR; Subkey: ".aac\OpenWithProgids"; ValueType: string; ValueName: "AudioTranscriptor.audio"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc_audio
Root: HKCR; Subkey: ".wma\OpenWithProgids"; ValueType: string; ValueName: "AudioTranscriptor.audio"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc_audio
Root: HKCR; Subkey: ".mp4\OpenWithProgids"; ValueType: string; ValueName: "AudioTranscriptor.video"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc_video
Root: HKCR; Subkey: ".mkv\OpenWithProgids"; ValueType: string; ValueName: "AudioTranscriptor.video"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc_video
Root: HKCR; Subkey: ".mov\OpenWithProgids"; ValueType: string; ValueName: "AudioTranscriptor.video"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc_video
Root: HKCR; Subkey: ".avi\OpenWithProgids"; ValueType: string; ValueName: "AudioTranscriptor.video"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc_video
Root: HKCR; Subkey: ".webm\OpenWithProgids"; ValueType: string; ValueName: "AudioTranscriptor.video"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc_video
Root: HKCR; Subkey: ".m4v\OpenWithProgids"; ValueType: string; ValueName: "AudioTranscriptor.video"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc_video
