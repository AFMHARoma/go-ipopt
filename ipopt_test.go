package ipopt

import "testing"

type MyProblem struct {
	g_offset [2]float32
	problem  *Problem
}

func (p *MyProblem) eval_f(x []float32, _ bool, objValue []float32) bool {
	objValue[0] = x[0]*x[3]*(x[0]+x[1]+x[2]) + x[2]
	return true
}

func (p *MyProblem) eval_grad_f(x []float32, _ bool, grad []float32) bool {
	grad[0] = x[0]*x[3] + x[3]*(x[0]+x[1]+x[2])
	grad[1] = x[0] * x[3]
	grad[2] = x[0]*x[3] + 1
	grad[3] = x[0] * (x[0] + x[1] + x[2])
	return true
}

func (p *MyProblem) eval_g(x []float32, _ bool, _ int, g []float32) bool {

	g[0] = x[0]*x[1]*x[2]*x[3] + p.g_offset[0]
	g[1] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3] + p.g_offset[1]

	return true
}

func (p *MyProblem) eval_jac_g(x []float32, _ bool, _ int, jac [2][]int32, values []float32) bool {
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
		values[0] = x[1] * x[2] * x[3] /* 0,0 */
		values[1] = x[0] * x[2] * x[3] /* 0,1 */
		values[2] = x[0] * x[1] * x[3] /* 0,2 */
		values[3] = x[0] * x[1] * x[2] /* 0,3 */

		values[4] = 2 * x[0] /* 1,0 */
		values[5] = 2 * x[1] /* 1,1 */
		values[6] = 2 * x[2] /* 1,2 */
		values[7] = 2 * x[3] /* 1,3 */
	}
	return true
}

func (p *MyProblem) eval_h(x []float32, _ bool, objFactor float32, _ int, lambda []float32, _ bool, hess [2][]int32, values []float32) bool {
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
		values[0] = objFactor * (2 * x[3]) /* 0,0 */

		values[1] = objFactor * (x[3]) /* 1,0 */
		values[2] = 0                  /* 1,1 */

		values[3] = objFactor * (x[3]) /* 2,0 */
		values[4] = 0                  /* 2,1 */
		values[5] = 0                  /* 2,2 */

		values[6] = objFactor * (2*x[0] + x[1] + x[2]) /* 3,0 */
		values[7] = objFactor * (x[0])                 /* 3,1 */
		values[8] = objFactor * (x[0])                 /* 3,2 */
		values[9] = 0                                  /* 3,3 */

		values[1] += lambda[0] * (x[2] * x[3]) /* 1,0 */

		values[3] += lambda[0] * (x[1] * x[3]) /* 2,0 */
		values[4] += lambda[0] * (x[0] * x[3]) /* 2,1 */

		values[6] += lambda[0] * (x[1] * x[2]) /* 3,0 */
		values[7] += lambda[0] * (x[0] * x[2]) /* 3,1 */
		values[8] += lambda[0] * (x[0] * x[1]) /* 3,2 */

		values[0] += lambda[1] * 2 /* 0,0 */

		values[2] += lambda[1] * 2 /* 1,1 */

		values[5] += lambda[1] * 2 /* 2,2 */

		values[9] += lambda[1] * 2 /* 3,3 */
	}
	return true
}

func TestVersion(t *testing.T) {
	n := 4
	x_L := make([]float32, n)
	x_U := make([]float32, n)

	for i := 0; i < n; i++ {
		x_L[i] = 1.0
		x_U[i] = 5.0
	}

	m := 2
	g_L := make([]float32, m)
	g_U := make([]float32, m)

	g_L[0] = 25
	g_U[0] = 2e19
	g_L[1] = 40
	g_U[1] = 40

	nele_jac := 8
	nele_hess := 10

	p := &MyProblem{}

	opt := ProblemOptions{
		Variables:              [2][]float32{x_L, x_U},
		Constraints:            [2][]float32{g_L, g_U},
		NumConstraintJacobian:  nele_jac,
		NumHessianOfLagrangian: nele_hess,
		Eval:                   p.eval_f,
		EvalGrad:               p.eval_grad_f,
		EvalG:                  p.eval_g,
		EvalJacG:               p.eval_jac_g,
		EvalH:                  p.eval_h,
	}

	problem, err := NewProblem(opt)
	if err != nil {
		t.Fail()
	}

	problem.AddNumOption("tol", 3.82e-6)
	problem.AddStrOption("mu_strategy", "adaptive")
	problem.AddStrOption("output_file", "ipopt.out")

	p.problem = problem
	p.g_offset[0] = 0.
	p.g_offset[1] = 0.

	x := make([]float32, n)
	x[0] = 1.0
	x[1] = 5.0
	x[2] = 5.0
	x[3] = 1.0

	mult_g := make([]float32, m)
	mult_x_L := make([]float32, n)
	mult_x_U := make([]float32, n)

	objVal := []float32{0}

	objVal, status := p.problem.Solve(x, nil, objVal, mult_g, mult_x_L, mult_x_U)

	if status == nil {

	}
	p.g_offset[0] = 0.2
	problem.AddStrOption("warm_start_init_point", "yes")
	problem.AddNumOption("bound_push", 1e-5)
	problem.AddNumOption("bound_frac", 1e-5)

	objVal, status = p.problem.Solve(x, nil, objVal, mult_g, mult_x_L, mult_x_U)
	if status == nil {

	}
}
