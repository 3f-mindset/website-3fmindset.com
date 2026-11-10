$ErrorActionPreference = "Stop"

$ListenPort = 8080
$ConnectAddress = "127.0.0.1"
$ConnectPort = 8080
$RuleName = "Allow Hugo Preview 8080"

Write-Host "Configuring Windows exposure for Hugo preview on port $ListenPort..."

netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=$ListenPort | Out-Null
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=$ListenPort connectaddress=$ConnectAddress connectport=$ConnectPort

netsh advfirewall firewall delete rule name="$RuleName" | Out-Null
netsh advfirewall firewall add rule name="$RuleName" dir=in action=allow protocol=TCP localport=$ListenPort | Out-Null

Write-Host ""
Write-Host "Done."
Write-Host "Portproxy:"
netsh interface portproxy show v4tov4
Write-Host ""
Write-Host "Firewall rule:"
netsh advfirewall firewall show rule name="$RuleName"
