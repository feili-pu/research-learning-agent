@echo off
setlocal

cd /d "%~dp0.."

if not exist ".env" (
  echo .env file was not found.
  exit /b 1
)

for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
  if not "%%A"=="" set "%%A=%%B"
)

echo has_key: configured
echo base_url: %RLA_OPENAI_BASE_URL%
echo model: %RLA_LLM_MODEL%
echo wire_api: %RLA_OPENAI_WIRE_API%
echo.

if /i "%RLA_OPENAI_WIRE_API%"=="chat" (
  curl -sS -X POST "%RLA_OPENAI_BASE_URL%/chat/completions" ^
    -H "Authorization: Bearer %OPENAI_API_KEY%" ^
    -H "Content-Type: application/json" ^
    -d "{\"model\":\"%RLA_LLM_MODEL%\",\"messages\":[{\"role\":\"user\",\"content\":\"用中文回答：接口测试成功了吗？\"}]}"
) else (
  curl -sS -X POST "%RLA_OPENAI_BASE_URL%/responses" ^
  -H "Authorization: Bearer %OPENAI_API_KEY%" ^
  -H "Content-Type: application/json" ^
    -d "{\"model\":\"%RLA_LLM_MODEL%\",\"input\":[{\"role\":\"user\",\"content\":\"用中文回答：接口测试成功了吗？\"}]}"
)

echo.
