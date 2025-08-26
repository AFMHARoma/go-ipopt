package ipopt

import (
	"fmt"

	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/mat"

	"testing"
)

// ComputeHessian численно оценивает гессиан функции f в точке x.
func ComputeHessian(f func(x []float64) float64, x []float64) *mat.SymDense {
	n := len(x)                         // Размерность входного вектора
	hessianData := make([]float64, n*n) // Массив для хранения значений гессиана

	// Обрабатываем каждую переменную для вычисления вторых производных
	for i := range n {
		// Для каждой переменной вычисляем частные производные (градиент)
		gradFunc := func(xi []float64) float64 {
			g := fd.Gradient(nil, f, xi, nil) // Вычисляем градиент функции f по x
			return g[i]                       // Возвращаем i-ю компоненту градиента
		}

		// Вычисляем вторую производную (гессиан) для этой компоненты
		secondGrad := fd.Gradient(nil, gradFunc, x, nil)

		// Записываем результаты в матрицу гессиана
		for j := range n {
			hessianData[i*n+j] = secondGrad[j] // Записываем в ячейку гессиана
		}
	}

	// Создаём симметричную матрицу гессиана
	return mat.NewSymDense(n, hessianData)
}

// ComputeLagrangianHessian строит гессиан лагранжиана с учётом весов objFactor и множителей lambda.
func ComputeLagrangianHessian(
	objFunc func(x []float64) float64,
	constrFunc func(y, x []float64),
	objFactor float64,
	lambda []float64,
	x []float64,
) *mat.SymDense {
	n := len(x)
	hessianData := make([]float64, n*n)

	// Вычисляем гессиан целевой функции
	if objFactor != 0 {
		hessObj := ComputeHessian(objFunc, x)

		for i := 0; i < n; i++ {
			for j := 0; j <= i; j++ {
				v := hessObj.At(i, j) * objFactor

				hessianData[i*n+j] += v

				if i != j {
					hessianData[j*n+i] += v
				}
			}
		}
	}

	// Для ограничений — через градиенты constraintFunc
	if lambda != nil {
		m := len(lambda)

		for k := 0; k < m; k++ {
			constraintGradFunc := func(xi []float64) float64 {
				y := make([]float64, m)
				constrFunc(y, xi)

				return y[k]
			}
			hessGk := ComputeHessian(constraintGradFunc, x)

			for i := 0; i < n; i++ {
				for j := 0; j <= i; j++ {
					v := hessGk.At(i, j) * lambda[k]

					hessianData[i*n+j] += v

					if i != j {
						hessianData[j*n+i] += v
					}
				}
			}
		}
	}

	return mat.NewSymDense(n, hessianData)
}

// ExtractUpperTriangle извлекает элементы верхнего треугольника симметричной матрицы.
func ExtractUpperTriangle(sym *mat.SymDense) []float64 {
	n := sym.SymmetricDim()

	result := make([]float64, 0, n*(n+1)/2)

	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			result = append(result, sym.At(i, j))
		}
	}

	return result
}

func ConvFloat32ArrToFloat64(x []float32) []float64 {
	newArray := make([]float64, len(x))

	for i, r := range x {
		newArray[i] = float64(r)
	}

	return newArray
}

type MyProblem struct {
	problem *Problem
}

func (p *MyProblem) targetFunc(x []float64) float64 {
	return x[0]*x[3]*(x[0]+x[1]+x[2]) + x[2]
}

func (p *MyProblem) evalF(x []float64, _ bool, objValue *float64) bool {
	res := p.targetFunc(x)

	*objValue = res

	return true
}

func (p *MyProblem) evalGradF(x []float64, _ bool, grad []float64) bool {
	new_grad := fd.Gradient(
		nil,
		p.targetFunc,
		x,
		nil,
	)

	for i := 0; i < len(grad); i++ {
		grad[i] = new_grad[i]
	}

	return true
}

func (p *MyProblem) evalGVector(y, x []float64) {
	y[0] = x[0] * x[1] * x[2] * x[3]
	y[1] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]
}

// evalG реализует интерфейс IPOPT: значения ограничений g(x).
func (p *MyProblem) evalG(x []float64, _ bool, _ int, g []float64) bool {
	g[0] = x[0] * x[1] * x[2] * x[3]
	g[1] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]

	return true
}

// evalJacG реализует интерфейс IPOPT: якобиан ограничений (структура и значения).
func (p *MyProblem) evalJacG(x []float64, _ bool, _ int, jac [2][]int32, values []float64) bool {
	dense := mat.NewDense(2, 4, nil)

	if values == nil {
		jac[0][0] = 0
		jac[1][0] = 0
		jac[0][1] = 0
		jac[1][1] = 1
		jac[0][2] = 0
		jac[1][2] = 2
		jac[0][3] = 0
		jac[1][3] = 3
		jac[0][4] = 1
		jac[1][4] = 0
		jac[0][5] = 1
		jac[1][5] = 1
		jac[0][6] = 1
		jac[1][6] = 2
		jac[0][7] = 1
		jac[1][7] = 3
	} else {
		fd.Jacobian(dense, p.evalGVector, x, nil)
		matrix := dense.RawMatrix().Data

		for i := 0; i < len(matrix); i++ {
			values[i] = matrix[i]
		}
	}

	return true
}

// evalH реализует интерфейс IPOPT: гессиан лагранжиана (структура и значения).
func (p *MyProblem) evalH(
	x []float64,
	_ bool,
	objFactor float64,
	_ int,
	lambda []float64,
	_ bool,
	hess [2][]int32,
	values []float64,
) bool {
	if values == nil {
		idx := 0
		for row := 0; row < 4; row++ {
			for col := 0; col <= row; col++ {
				hess[0][idx] = int32(row)
				hess[1][idx] = int32(col)
				idx++
			}
		}
	} else {
		hess := ComputeLagrangianHessian(p.targetFunc, p.evalGVector, objFactor, lambda, x)
		matrix := ExtractUpperTriangle(hess)

		for i := 0; i < len(matrix); i++ {
			values[i] = matrix[i]
		}
	}

	return true
}

func TestVersion(t *testing.T) {
	n := 4
	x_L := make([]float64, n)
	x_U := make([]float64, n)

	for i := 0; i < n; i++ {
		x_L[i] = 1.0
		x_U[i] = 5.0
	}

	m := 2
	g_L := make([]float64, m)
	g_U := make([]float64, m)

	g_L[0] = 25
	g_U[0] = 2e19
	g_L[1] = 40
	g_U[1] = 40

	nele_jac := 8
	nele_hess := 10

	p := &MyProblem{}

	opt := ProblemOptions{
		Variables:              [2][]float64{x_L, x_U},
		Constraints:            [2][]float64{g_L, g_U},
		NumConstraintJacobian:  nele_jac,
		NumHessianOfLagrangian: nele_hess,
		Eval:                   p.evalF,
		EvalGrad:               p.evalGradF,
		EvalG:                  p.evalG,
		EvalJacG:               p.evalJacG,
		EvalH:                  p.evalH,
	}

	problem, err := NewProblem(opt)
	if err != nil {
		t.Fail()
	}

	problem.AddNumOption("tol", 3.82e-6)
	problem.AddStrOption("mu_strategy", "adaptive")

	p.problem = problem

	x := make([]float64, n)
	x[0] = 1.0
	x[1] = 5.0
	x[2] = 5.0
	x[3] = 1.0

	mult_g := make([]float64, m)
	mult_x_L := make([]float64, n)
	mult_x_U := make([]float64, n)

	objVal := []float64{0}

	objVal, status := p.problem.Solve(x, nil, objVal, mult_g, mult_x_L, mult_x_U, true)
	if status == nil {

	}

	fmt.Println(x)

	problem.AddStrOption("warm_start_init_point", "yes")
	problem.AddNumOption("bound_push", 1e-5)
	problem.AddNumOption("bound_frac", 1e-5)

	objVal, status = p.problem.Solve(x, nil, objVal, mult_g, mult_x_L, mult_x_U, true)
	if status == nil {

	}

	fmt.Println(x)
}
