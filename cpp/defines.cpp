
#include "defines.h"
#define str_macro(X) #X
#define str(X) str_macro(X)

namespace libtab
{
std::string version() { return str(LIBTAB_VERSION); }
}; // namespace libtab
