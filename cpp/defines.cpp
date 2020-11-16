
#include "defines.h"
#define str_macro(X) #X
#define str(X) str_macro(X)

namespace libtab
{
/// Return the version number of libtab to check compatibility if using
/// across projects
/// @return version string
std::string version() { return str(LIBTAB_VERSION); }
}; // namespace libtab
