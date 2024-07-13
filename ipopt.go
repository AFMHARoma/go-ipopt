package ipopt

/*
#include "ipopt_c_api.h"
#cgo linux CFLAGS:-I ./lib
#cgo darwin CFLAGS:-I ./lib
#cgo darwin,arm CFLAGS:-I ./lib
#cgo windows CFLAGS:-I ./lib
#cgo linux CXXFLAGS: -I ./lib -std=c++14
#cgo darwin CXXFLAGS: -I ./lib -std=gnu++14
#cgo darwin,arm CXXFLAGS: -I ./lib -std=gnu++14
#cgo windows CXXFLAGS: -I ./lib -std=c++14
#cgo linux LDFLAGS: -L ./lib/linux  -Wl,--start-group -lstdc++ -lipopt -llapack -lblas -lma27 -lmetis -ldl -lm -lcipopt -Wl,--end-group
#cgo darwin LDFLAGS: -L /usr/lib -lc++ -L ./lib/darwin -lipopt -lcipopt -llapack -lblas -lma27 -lmetis -lm
#cgo darwin,arm LDFLAGS: -L /usr/lib -lc++ -L ./lib/darwin_arm  -lipopt  -llapack -lblas -lma27 -lmetis -lcipopt -lm
#cgo windows LDFLAGS: -L ./lib/windows -lipopt -llapack -lblas -lma27 -lmetis -lcipopt -fPIC
*/
import "C"
