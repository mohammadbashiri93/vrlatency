call set_secrets.bat

@echo off
set /p DELETE_ALL_DATA="Delete existing files (y/n): "
IF %DELETE_ALL_DATA%==y (
   call rmdir data /s /Q
)

call mkdir data
call .\vendor\pscp\pscp -pw %THETA_PASSWD% %THETA_USER%@%THETA_IP%:/storage/%THETA_USER%/vrlatency/*.csv .\data