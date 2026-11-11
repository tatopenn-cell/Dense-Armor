; Setup wizard of the Dense-Armor dashboard (Inno Setup 6).
; Build: iscc /DStage=<folder with the files> /DOutDir=<folder for the exe> /DAppVersion=1.5.0 dense-armor.iss
#ifndef AppVersion
  #define AppVersion "1.5.0"
#endif
#ifndef Stage
  #define Stage "..\..\..\stage"
#endif
#ifndef OutDir
  #define OutDir "..\..\..\out"
#endif

[Setup]
AppId={{B7E1B0E0-6C27-4E58-9E1B-DA0D2A6C5A11}
AppName=Dense-Armor
AppVersion={#AppVersion}
AppPublisher=Salvatore Pennacchio
AppPublisherURL=https://github.com/tatopenn-cell/Dense-Armor
AppSupportURL=https://github.com/tatopenn-cell/Dense-Armor/issues
DefaultDirName={localappdata}\Programs\Dense-Armor
DefaultGroupName=Dense-Armor
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
LicenseFile={#Stage}\LICENSE
SetupIconFile={#Stage}\tools\dashboard\dense-armor.ico
UninstallDisplayIcon={app}\tools\dashboard\dense-armor.ico
UninstallDisplayName=Dense-Armor
OutputDir={#OutDir}
OutputBaseFilename=Dense-Armor-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "italian"; MessagesFile: "compiler:Languages\Italian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#Stage}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autodesktop}\Dense-Armor"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\tools\dashboard\launcher.py"""; WorkingDir: "{app}\tools\dashboard"; IconFilename: "{app}\tools\dashboard\dense-armor.ico"; Tasks: desktopicon
Name: "{autoprograms}\Dense-Armor\Dense-Armor"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\tools\dashboard\launcher.py"""; WorkingDir: "{app}\tools\dashboard"; IconFilename: "{app}\tools\dashboard\dense-armor.ico"
Name: "{autoprograms}\Dense-Armor\{cm:UninstallProgram,Dense-Armor}"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\tools\dashboard\launcher.py"""; WorkingDir: "{app}\tools\dashboard"; Description: "{cm:LaunchProgram,Dense-Armor}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -Command ""Get-Process | Where-Object {{ $_.Path -and $_.Path.StartsWith('{app}', 'OrdinalIgnoreCase') }} | Stop-Process -Force"""; Flags: runhidden; RunOnceId: "StopDenseArmor"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\tools\dashboard\run"
Type: filesandordirs; Name: "{app}\dense_armor.egg-info"
Type: files; Name: "{app}\install.log"

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  rc: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    WizardForm.StatusLabel.Caption := 'Installing Python and the packages (a few minutes the first time) ...';
    WizardForm.FileNameLabel.Caption := '';
    if (not Exec('powershell.exe',
      '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\tools\dashboard\installer\setup-env.ps1') + '" -App "' + ExpandConstant('{app}') + '"',
      '', SW_HIDE, ewWaitUntilTerminated, rc)) or (rc <> 0) then
      RaiseException('The packages could not be installed. The log is in ' + ExpandConstant('{app}\install.log'));
  end;
end;
