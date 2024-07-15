package ipopt

// #include <stdbool.h>
import "C"
import (
	"math"
	"unsafe"
)

//export evalFunc
func evalFunc(n C.int, x *C.float, newX C.bool, objValue *C.float, userData unsafe.Pointer) C.bool {
	p := (*problemCallback)(userData)
	if p.eval != nil {
		var goX []float32
		if x == nil {
			goX = nil
		} else {
			goX = (*[(math.MaxInt32 - 1) / unsafe.Sizeof(*x)]float32)(unsafe.Pointer(x))[:n:n]
		}
		goobjV := (*[((math.MaxInt32 - 1) / unsafe.Sizeof(*objValue))]float32)(unsafe.Pointer(objValue))[:n:n]
		return (C.bool)(p.eval(goX, (bool)(newX), goobjV))
	}
	return false
}

//export evalGradFunc
func evalGradFunc(n C.int, x *C.float, newX C.bool, grad *C.float, userData unsafe.Pointer) C.bool {
	p := (*problemCallback)(userData)
	if p.evalGrad != nil {
		var goX []float32
		if x == nil {
			goX = nil
		} else {
			goX = (*[(math.MaxInt32 - 1) / unsafe.Sizeof(*x)]float32)(unsafe.Pointer(x))[:n:n]
		}
		goGrad := (*[((math.MaxInt32 - 1) / unsafe.Sizeof(*grad))]float32)(unsafe.Pointer(grad))[:n:n]
		return (C.bool)(p.evalGrad(goX, (bool)(newX), goGrad))
	}
	return false
}

//export evalGFunc
func evalGFunc(n C.int, x *C.float, newX C.bool, m C.int, g *C.float, userData unsafe.Pointer) C.bool {
	p := (*problemCallback)(userData)
	if p.evalG != nil {
		var goX []float32
		if x == nil {
			goX = nil
		} else {
			goX = (*[(math.MaxInt32 - 1) / unsafe.Sizeof(*x)]float32)(unsafe.Pointer(x))[:n:n]
		}
		gog := (*[((math.MaxInt32 - 1) / unsafe.Sizeof(*g))]float32)(unsafe.Pointer(g))[:m:m]
		return (C.bool)(p.evalG(goX, (bool)(newX), (int)(m), gog))
	}
	return false
}

//export evalJacGFunc
func evalJacGFunc(n C.int, x *C.float, newX C.bool, m C.int, nele_jac C.int, iRow *C.int, jCol *C.int, values *C.float, userData unsafe.Pointer) C.bool {
	p := (*problemCallback)(userData)
	if p.evalJacG != nil {
		var goX []float32
		if x == nil {
			goX = nil
		} else {
			goX = (*[(math.MaxInt32 - 1) / unsafe.Sizeof(*x)]float32)(unsafe.Pointer(x))[:n:n]
		}

		var govalues []float32
		if values != nil {
			govalues = (*[((math.MaxInt32 - 1) / unsafe.Sizeof(*values))]float32)(unsafe.Pointer(values))[:nele_jac:nele_jac]
		} else {
			govalues = nil
		}

		var goiRow []int32
		if iRow != nil {
			goiRow = (*[((math.MaxInt32 - 1) / unsafe.Sizeof(*iRow))]int32)(unsafe.Pointer(iRow))[:nele_jac:nele_jac]
		} else {
			goiRow = nil
		}

		var gojCol []int32
		if jCol != nil {
			gojCol = (*[((math.MaxInt32 - 1) / unsafe.Sizeof(*jCol))]int32)(unsafe.Pointer(jCol))[:nele_jac:nele_jac]
		} else {
			gojCol = nil
		}

		jac := [2][]int32{goiRow, gojCol}
		return (C.bool)(p.evalJacG(goX, (bool)(newX), (int)(m), jac, govalues))
	}
	return false
}

//export evalHFunc
func evalHFunc(n C.int, x *C.float, newX C.bool, objFactor C.float, m C.int, lambda *C.float, newLambda C.bool, nele_hess C.int, iRow *C.int, jCol *C.int, values *C.float, userData unsafe.Pointer) C.bool {
	p := (*problemCallback)(userData)
	if p.evalH != nil {
		var goX []float32
		if x == nil {
			goX = nil
		} else {
			goX = (*[(math.MaxInt32 - 1) / unsafe.Sizeof(*x)]float32)(unsafe.Pointer(x))[:n:n]
		}
		golambda := (*[((math.MaxInt32 - 1) / unsafe.Sizeof(*lambda))]float32)(unsafe.Pointer(lambda))[:m:m]
		var govalues []float32
		if values != nil {
			govalues = (*[((math.MaxInt32 - 1) / unsafe.Sizeof(*values))]float32)(unsafe.Pointer(values))[:nele_hess:nele_hess]
		} else {
			govalues = nil
		}

		var goiRow []int32
		if iRow != nil {
			goiRow = (*[((math.MaxInt32 - 1) / unsafe.Sizeof(*iRow))]int32)(unsafe.Pointer(iRow))[:nele_hess:nele_hess]
		} else {
			goiRow = nil
		}

		var gojCol []int32
		if jCol != nil {
			gojCol = (*[((math.MaxInt32 - 1) / unsafe.Sizeof(*jCol))]int32)(unsafe.Pointer(jCol))[:nele_hess:nele_hess]
		} else {
			gojCol = nil
		}

		hess := [2][]int32{goiRow, gojCol}
		return (C.bool)(p.evalH(goX, (bool)(newX), (float32)(objFactor), (int)(m), golambda, (bool)(newLambda), hess, govalues))
	}
	return false
}
