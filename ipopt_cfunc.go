package ipopt

// #include <stdbool.h>
import "C"
import (
	"math"
	"unsafe"
)

//export evalFunc
func evalFunc(n C.int, x *C.double, newX C.bool, objValue *C.double, userData unsafe.Pointer) C.bool {
	p := (*problemCallback)(userData)
	if p.eval != nil {
		var goX []float64
		if x == nil {
			goX = nil
		} else {
			goX = (*[(math.MaxInt32 - 1) / unsafe.Sizeof(*x)]float64)(unsafe.Pointer(x))[:n:n]
		}

		// objValue — это одно число, а не массив
		goObjValue := (*float64)(unsafe.Pointer(objValue))

		return (C.bool)(p.eval(goX, bool(newX), goObjValue))
	}
	return false
}

//export evalGradFunc
func evalGradFunc(n C.int, x *C.double, newX C.bool, grad *C.double, userData unsafe.Pointer) C.bool {
	p := (*problemCallback)(userData)
	if p.evalGrad != nil {
		var goX []float64
		if x == nil {
			goX = nil
		} else {
			goX = (*[(math.MaxInt32 - 1) / unsafe.Sizeof(*x)]float64)(unsafe.Pointer(x))[:n:n]
		}

		var goGrad []float64
		if grad == nil {
			goGrad = nil
		} else {
			goGrad = (*[(math.MaxInt32 - 1) / unsafe.Sizeof(*grad)]float64)(unsafe.Pointer(grad))[:n:n]
		}

		return (C.bool)(p.evalGrad(goX, bool(newX), goGrad))
	}
	return false
}

//export evalGFunc
func evalGFunc(n C.int, x *C.double, newX C.bool, m C.int, g *C.double, userData unsafe.Pointer) C.bool {
	p := (*problemCallback)(userData)
	if p.evalG != nil {
		var goX []float64
		if x == nil {
			goX = nil
		} else {
			goX = (*[(math.MaxInt32 - 1) / unsafe.Sizeof(*x)]float64)(unsafe.Pointer(x))[:n:n]
		}

		var gog []float64
		if g == nil {
			gog = nil
		} else {
			gog = (*[(math.MaxInt32 - 1) / unsafe.Sizeof(*g)]float64)(unsafe.Pointer(g))[:m:m]
		}

		return (C.bool)(p.evalG(goX, bool(newX), int(m), gog))
	}
	return false
}

//export evalJacGFunc
func evalJacGFunc(n C.int, x *C.double, newX C.bool, m C.int, nele_jac C.int, iRow *C.int, jCol *C.int, values *C.double, userData unsafe.Pointer) C.bool {
	p := (*problemCallback)(userData)
	if p.evalJacG != nil {
		var goX []float64
		if x == nil {
			goX = nil
		} else {
			goX = (*[(math.MaxInt32 - 1) / unsafe.Sizeof(*x)]float64)(unsafe.Pointer(x))[:n:n]
		}

		var govalues []float64
		if values != nil {
			govalues = (*[(math.MaxInt32 - 1) / unsafe.Sizeof(*values)]float64)(unsafe.Pointer(values))[:nele_jac:nele_jac]
		} else {
			govalues = nil
		}

		var goiRow []int32
		if iRow != nil {
			goiRow = (*[(math.MaxInt32 - 1) / unsafe.Sizeof(*iRow)]int32)(unsafe.Pointer(iRow))[:nele_jac:nele_jac]
		} else {
			goiRow = nil
		}

		var gojCol []int32
		if jCol != nil {
			gojCol = (*[(math.MaxInt32 - 1) / unsafe.Sizeof(*jCol)]int32)(unsafe.Pointer(jCol))[:nele_jac:nele_jac]
		} else {
			gojCol = nil
		}

		jac := [2][]int32{goiRow, gojCol}
		return (C.bool)(p.evalJacG(goX, bool(newX), int(m), jac, govalues))
	}
	return false
}

//export evalHFunc
func evalHFunc(n C.int, x *C.double, newX C.bool, objFactor C.double, m C.int, lambda *C.double, newLambda C.bool, nele_hess C.int, iRow *C.int, jCol *C.int, values *C.double, userData unsafe.Pointer) C.bool {
	p := (*problemCallback)(userData)
	if p.evalH != nil {
		var goX []float64
		if x == nil {
			goX = nil
		} else {
			goX = (*[(math.MaxInt32 - 1) / unsafe.Sizeof(*x)]float64)(unsafe.Pointer(x))[:n:n]
		}

		var golambda []float64
		if lambda == nil {
			golambda = nil
		} else {
			golambda = (*[(math.MaxInt32 - 1) / unsafe.Sizeof(*lambda)]float64)(unsafe.Pointer(lambda))[:m:m]
		}

		var govalues []float64
		if values != nil {
			govalues = (*[(math.MaxInt32 - 1) / unsafe.Sizeof(*values)]float64)(unsafe.Pointer(values))[:nele_hess:nele_hess]
		} else {
			govalues = nil
		}

		var goiRow []int32
		if iRow != nil {
			goiRow = (*[(math.MaxInt32 - 1) / unsafe.Sizeof(*iRow)]int32)(unsafe.Pointer(iRow))[:nele_hess:nele_hess]
		} else {
			goiRow = nil
		}

		var gojCol []int32
		if jCol != nil {
			gojCol = (*[(math.MaxInt32 - 1) / unsafe.Sizeof(*jCol)]int32)(unsafe.Pointer(jCol))[:nele_hess:nele_hess]
		} else {
			gojCol = nil
		}

		hess := [2][]int32{goiRow, gojCol}
		return (C.bool)(p.evalH(goX, bool(newX), float64(objFactor), int(m), golambda, bool(newLambda), hess, govalues))
	}
	return false
}
