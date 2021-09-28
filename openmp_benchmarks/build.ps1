If(!(test-path build))
{
    New-Item -Path "." -Name "build" -ItemType "directory"
}
Set-Location -Path "build"
Invoke-Expression "cmake -DCMAKE_BUILD_TYPE=Debug .."
#Invoke-Expression "make"
Invoke-Expression "devenv /build Debug .\OpenMPBench.sln"
Set-Location -Path ".."