@echo off
echo Starting Investment Risk Profiler...
echo.
echo Starting Backend Server...
start cmd /k "cd backend && python app.py"
echo Backend started on http://localhost:5000
echo.
echo Starting Frontend Server...
start cmd /k "cd frontend && npm start"
echo Frontend will start on http://localhost:3000
echo.
echo Both services are starting up...
echo Wait a moment for them to fully load, then open http://localhost:3000 in your browser
pause 