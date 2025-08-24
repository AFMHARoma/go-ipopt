package ipopt

/*
#include <stdlib.h>
#include "ipopt_c_api.h"
#cgo linux CFLAGS:-I ./lib
#cgo darwin CFLAGS:-I ./lib
#cgo darwin,arm CFLAGS:-I ./lib
#cgo windows CFLAGS:-I ./lib
#cgo linux CXXFLAGS: -I ./lib -std=c++14
#cgo darwin CXXFLAGS: -I ./lib -std=gnu++14
#cgo darwin,arm CXXFLAGS: -I ./lib -std=gnu++14
#cgo windows CXXFLAGS: -I ./lib -std=c++14
#cgo linux LDFLAGS: -L ./lib/linux  -Wl,--start-group -lstdc++ -lipopt -llapack -lblas -lma27 -lmetis -ldl -lm -lcipopt -lgfortran -Wl,--end-group
#cgo darwin LDFLAGS: -L /usr/local/gfortran/lib -Wl,-rpath,/usr/local/gfortran/lib
#cgo darwin,amd64 LDFLAGS: -L /usr/lib -lc++ -L ./lib/darwin -lipopt -lcipopt   -lma27 -lmetis -lm  -framework Accelerate  -lgfortran
#cgo darwin,arm64 LDFLAGS: -L /usr/lib -lc++ -L ./lib/darwin_arm  -lipopt  -lma27 -lmetis -lcipopt -lm  -framework Accelerate  -lgfortran
#cgo windows LDFLAGS: -L ./lib/windows -lipopt -llapack -lblas -lma27 -lmetis -lcipopt -fPIC

extern bool evalFunc(int n, float *x, bool new_x, float *obj_value,
                          void *user_data);
extern bool evalGradFunc(int n, float *x, bool new_x, float *grad_f,
                               void *user_data);
extern bool evalGFunc(int n, float *x, bool new_x, int m, float *g,
                          void *user_data);
extern bool evalJacGFunc(int n, float *x, bool new_x, int m, int nele_jac,
                              int *iRow, int *jCol, float *values,
                              void *user_data);
extern bool evalHFunc(int n, float *x, bool new_x, float obj_factor, int m,
                          float *lambda, bool new_lambda, int nele_hess,
                          int *iRow, int *jCol, float *values, void *user_data);

bool ipopt_eval_func_go(int n, float *x, bool new_x, float *obj_value,
                          void *user_data) {
    return evalFunc(n, x, new_x, obj_value, user_data);
}

bool ipopt_eval_grad_func_go(int n, float *x, bool new_x, float *grad_f,
                               void *user_data) {
    return evalGradFunc(n, x, new_x, grad_f, user_data);
}

bool ipopt_eval_g_func_go(int n, float *x, bool new_x, int m, float *g,
                          void *user_data) {
    return evalGFunc(n, x, new_x, m, g, user_data);
}

bool ipopt_eval_jac_g_func_go(int n, float *x, bool new_x, int m, int nele_jac,
                              int *iRow, int *jCol, float *values,
                              void *user_data) {
    return evalJacGFunc(n, x, new_x, m, nele_jac, iRow, jCol, values, user_data);
}

bool ipopt_eval_h_func_go(int n, float *x, bool new_x, float obj_factor, int m,
                          float *lambda, bool new_lambda, int nele_hess,
                          int *iRow, int *jCol, float *values, void *user_data) {
    return evalHFunc(n, x, new_x, obj_factor, m, lambda, new_lambda, nele_hess, iRow, jCol, values, user_data);
}
*/
import "C"
import (
	"errors"
	"unsafe"
)

const (
	IPOPT_SOLVE_SUCCEEDED                    = int(C.solve_succeeded)
	IPOPT_SOLVED_TO_ACCEPTABLE_LEVEL         = int(C.solved_to_acceptable_level)
	IPOPT_INFEASIBLE_PROBLEM_DETECTED        = int(C.infeasible_problem_detected)
	IPOPT_SEARCH_DIRECTION_BECOMES_TOO_SMALL = int(C.search_direction_becomes_too_small)
	IPOPT_DIVERGING_ITERATES                 = int(C.diverging_iterates)
	IPOPT_USER_REQUESTED_STOP                = int(C.user_requested_stop)
	IPOPT_FEASIBLE_POINT_FOUND               = int(C.feasible_point_found)
	IPOPT_MAXIMUM_ITERATIONS_EXCEEDED        = int(C.maximum_iterations_exceeded)
	IPOPT_RESTORATION_FAILED                 = int(C.restoration_failed)
	IPOPT_ERROR_IN_STEP_COMPUTATION          = int(C.error_in_step_computation)
	IPOPT_MAXIMUM_CPUTIME_EXCEEDED           = int(C.maximum_cputime_exceeded)
	IPOPT_MAXIMUM_WALLTIME_EXCEEDED          = int(C.maximum_walltime_exceeded)
	IPOPT_NOT_ENOUGH_DEGREES_OF_FREEDOM      = int(C.not_enough_degrees_of_freedom)
	IPOPT_INVALID_PROBLEM_DEFINITION         = int(C.invalid_problem_definition)
	IPOPT_INVALID_OPTION                     = int(C.invalid_option)
	IPOPT_INVALID_NUMBER_DETECTED            = int(C.invalid_number_detected)
	IPOPT_UNRECOVERABLE_EXCEPTION            = int(C.unrecoverable_exception)
	IPOPT_NON_IPOPT_EXCEPTION_THROWN         = int(C.non_ipopt_exception_thrown)
	IPOPT_INSUFFICIENT_MEMORY                = int(C.insufficient_memory)
	IPOPT_INTERNAL_ERROR                     = int(C.internal_error)
)

type EvalFunc func(x []float32, newX bool, objValue []float32) bool
type EvalGradFunc func(x []float32, newX bool, grad []float32) bool
type EvalGFunc func(x []float32, newX bool, m int, g []float32) bool
type EvalJacGFunc func(x []float32, newX bool, m int, jac [2][]int32, values []float32) bool
type EvalHFunc func(x []float32, newX bool, objFactor float32, m int, lambda []float32, newLambda bool, hess [2][]int32, values []float32) bool

type ProblemOptions struct {
	Variables              [2][]float32
	Constraints            [2][]float32
	NumConstraintJacobian  int
	NumHessianOfLagrangian int
	Eval                   EvalFunc
	EvalGrad               EvalGradFunc
	EvalG                  EvalGFunc
	EvalJacG               EvalJacGFunc
	EvalH                  EvalHFunc
}

type problemCallback struct {
	eval     EvalFunc
	evalGrad EvalGradFunc
	evalG    EvalGFunc
	evalJacG EvalJacGFunc
	evalH    EvalHFunc
}

type Problem struct {
	Inner *innerProblem
	opt   *ProblemOptions
}

type innerProblem struct {
	problem *C.struct__ipopt_problem_t
	cb      *problemCallback
}

func NewProblem(opt ProblemOptions) (*Problem, error) {
	var problem *C.struct__ipopt_problem_t

	if len(opt.Variables[0]) != len(opt.Variables[1]) {
		return nil, errors.New("variables len mast eq")
	}

	if len(opt.Constraints[0]) != len(opt.Constraints[1]) {
		return nil, errors.New("constraints len mast eq")
	}

	eval_f := (C.eval_f_cb)(unsafe.Pointer(C.ipopt_eval_func_go))
	eval_grad_f := (C.eval_grad_f_cb)(unsafe.Pointer(C.ipopt_eval_grad_func_go))
	eval_g := (C.eval_g_cb)(unsafe.Pointer(C.ipopt_eval_g_func_go))
	eval_jac_g := (C.eval_jac_g_cb)(unsafe.Pointer(C.ipopt_eval_jac_g_func_go))
	eval_h := (C.eval_h_cb)(unsafe.Pointer(C.ipopt_eval_h_func_go))

	xL := toCFloatArray(opt.Variables[0])
	xU := toCFloatArray(opt.Variables[1])

	gl := toCFloatArray(opt.Constraints[0])
	gu := toCFloatArray(opt.Constraints[1])

	n := len(opt.Variables[0])

	problem = C.ipopt_problem_create(C.int(n), &xL[0], &xU[0],
		C.int(len(opt.Constraints[0])), &gl[0], &gu[0],
		C.int(opt.NumConstraintJacobian), C.int(opt.NumHessianOfLagrangian),
		eval_f, eval_grad_f, eval_g, eval_jac_g, eval_h)

	cb := &problemCallback{
		eval:     opt.Eval,
		evalGrad: opt.EvalGrad,
		evalG:    opt.EvalG,
		evalJacG: opt.EvalJacG,
		evalH:    opt.EvalH,
	}

	g := &Problem{Inner: &innerProblem{
		problem: problem, cb: cb,
	}, opt: &opt}

	return g, nil
}

func (p *Problem) AddStrOption(param string, value string) {
	cparam := C.CString(param)
	cvalue := C.CString(value)
	C.ipopt_problem_add_str_option(p.Inner.problem, cparam, cvalue)
	C.free(unsafe.Pointer(cparam))
	C.free(unsafe.Pointer(cvalue))
}

func (p *Problem) AddIntOption(param string, value int) {
	cparam := C.CString(param)
	C.ipopt_problem_add_int_option(p.Inner.problem, cparam, C.int(value))
	C.free(unsafe.Pointer(cparam))
}

func (p *Problem) AddNumOption(param string, value float32) {
	cparam := C.CString(param)
	C.ipopt_problem_add_num_option(p.Inner.problem, cparam, C.float(value))
	C.free(unsafe.Pointer(cparam))
}

func (p *Problem) Solve(x []float32, g []float32, objVal []float32, multG []float32, multxL []float32, multxU []float32) ([]float32, error) {
	cX := toCFloatArray(x)
	cg := toCFloatArray(g)

	ccX := (*C.float)(&cX[0])

	var ccg *C.float
	if len(cg) > 0 {
		ccg = (*C.float)(&cg[0])
	} else {
		ccg = nil
	}

	cobjVal := toCFloatArray(objVal)
	cmultG := toCFloatArray(multG)

	cmultxL := toCFloatArray(multxL)
	cmultxU := toCFloatArray(multxU)

	userData := (*C.char)(unsafe.Pointer(p.Inner.cb))

	ret := (int)(C.ipopt_problem_solve(p.Inner.problem,
		ccX,
		ccg,
		(*C.float)(&cobjVal[0]),
		(*C.float)(&cmultG[0]),
		(*C.float)(&cmultxL[0]),
		(*C.float)(&cmultxU[0]),
		userData))

	toCopyFloatArray(cmultG, multG)
	toCopyFloatArray(cmultxL, multxL)
	toCopyFloatArray(cmultxU, multxU)
	toCopyFloatArray(cobjVal, objVal)
	toCopyFloatArray(cX, x)

	if ret == IPOPT_SOLVE_SUCCEEDED {
		return objVal, nil
	}
	return nil, resultStatus(ret)
}

func resultStatus(code int) error {
	var s string
	switch code {
	case IPOPT_SOLVED_TO_ACCEPTABLE_LEVEL:
		s = "Solved To Acceptable Level"
	case IPOPT_INFEASIBLE_PROBLEM_DETECTED:
		s = "Infeasible Problem Detected"
	case IPOPT_SEARCH_DIRECTION_BECOMES_TOO_SMALL:
		s = "Search Direction Becomes Too Small"
	case IPOPT_DIVERGING_ITERATES:
		s = "Diverging Iterates"
	case IPOPT_USER_REQUESTED_STOP:
		s = "User Requested Stop"
	case IPOPT_FEASIBLE_POINT_FOUND:
		s = "Feasible Point Found"
	case IPOPT_MAXIMUM_ITERATIONS_EXCEEDED:
		s = "Maximum Iterations Exceeded"
	case IPOPT_RESTORATION_FAILED:
		s = "Restoration Failed"
	case IPOPT_ERROR_IN_STEP_COMPUTATION:
		s = "Error In Step Computation"
	case IPOPT_MAXIMUM_CPUTIME_EXCEEDED:
		s = "Maximum CpuTime Exceeded"
	case IPOPT_MAXIMUM_WALLTIME_EXCEEDED:
		s = "Maximum WallTime Exceeded"
	case IPOPT_NOT_ENOUGH_DEGREES_OF_FREEDOM:
		s = "Not Enough Degrees Of Freedom"
	case IPOPT_INVALID_PROBLEM_DEFINITION:
		s = "Invalid Problem Definition"
	case IPOPT_INVALID_OPTION:
		s = "Invalid Option"
	case IPOPT_INVALID_NUMBER_DETECTED:
		s = "Invalid Number Detected"
	case IPOPT_UNRECOVERABLE_EXCEPTION:
		s = "Unrecoverable Exception"
	case IPOPT_NON_IPOPT_EXCEPTION_THROWN:
		s = "NonIpopt Exception Thrown"
	case IPOPT_INSUFFICIENT_MEMORY:
		s = "Insufficient Memory"
	case IPOPT_INTERNAL_ERROR:
		s = "Internal Error"
	}
	return errors.New(s)
}

func (p *innerProblem) free() {
	C.ipopt_problem_free(p.problem)
	p.problem = nil
}

func toCFloatArray(x []float32) []C.float {
	v := make([]C.float, len(x))
	for i := 0; i < len(x); i++ {
		v[i] = (C.float)(x[i])
	}
	return v
}

func toCopyFloatArray(srv []C.float, x []float32) []float32 {
	for i := 0; i < len(x); i++ {
		x[i] = (float32)(srv[i])
	}
	return x
}

func toGoFloatArray(x []C.float) []float32 {
	v := make([]float32, len(x))
	for i := 0; i < len(x); i++ {
		v[i] = float32(x[i])
	}
	return v
}

func toCIntArray(x []int) []C.int {
	v := make([]C.int, len(x))
	for i := 0; i < len(x); i++ {
		v[i] = (C.int)(x[i])
	}
	return v
}

func toGoIntArray(x []C.int) []int {
	v := make([]int, len(x))
	for i := 0; i < len(x); i++ {
		v[i] = int(x[i])
	}
	return v
}
