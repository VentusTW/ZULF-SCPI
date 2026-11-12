"""Program version: VERSION is set by hand (0.x while the program is being tested);
BUILD and BUILT_FOR are written by build_exe.ps1, which counts the builds of the .exe
of every version from 1 again. The build is never 0: before the first build of a
version it is 1, and that first build keeps 1."""

VERSION = "0.5.1"
BUILD = 5
BUILT_FOR = "0.5.1"


def version_text() -> str:
    """E.g. "0.5.0.3": the build is the fourth number."""
    return f"{VERSION}.{max(BUILD, 1)}"
