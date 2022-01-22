#include <bits/stdc++.h>
#include <pybind11/pybind11.h>

#define INF 100000000

struct EDGE {
    double cost;
	int cap, v;
	int next, re;
};
std::vector<EDGE> edge;
int e;
std::vector<int> head, vis;
double ans, cost, ans_flow;
int src, des, n;
void init(int MAXM, int MAXN) {
    edge.clear();
    edge.resize(MAXM);
    head.clear();
    head.resize(MAXN);
    vis.clear();
    vis.resize(MAXN);
	memset(&head[0], -1, sizeof(head[0]) * head.size());
	e = ans = cost = 0;
}
void addedge(int u, int v, int cap, double cost) {
	edge[e].v = v; edge[e].cap = cap;
	edge[e].cost = cost; edge[e].re = e + 1;
	edge[e].next = head[u]; head[u] = e++;
	edge[e].v = u; edge[e].cap = 0;
	edge[e].cost = -cost; edge[e].re = e - 1;
	edge[e].next = head[v]; head[v] = e++;
}
int aug(int u, int f) {//printf("%d %d|", u, f);
	if (u == des) {
		ans += cost * f;
        ans_flow += f;
		return f;
	}
	vis[u] = 1;
	int tmp = f;
	for (int i = head[u]; i != -1; i = edge[i].next) {
		if (edge[i].cap && !edge[i].cost && !vis[edge[i].v]) {
			int delta = aug(edge[i].v, tmp < edge[i].cap ? tmp :
				edge[i].cap);
			edge[i].cap -= delta; edge[edge[i].re].cap += delta;
			tmp -= delta;
			if (!tmp) return f;
		}
	}
	return f - tmp;
}
bool modlabel() {
	double delta = INF;
	for (int u = 0; u < n; u++) if (vis[u])
		for (int i = head[u]; i != -1; i = edge[i].next)
			if (edge[i].cap && !vis[edge[i].v] && edge[i].cost <
				delta) delta = edge[i].cost;
	if (delta == INF) return false;
	for (int u = 0; u < n; u++) if (vis[u])
		for (int i = head[u]; i != -1; i = edge[i].next)
			edge[i].cost -= delta, edge[edge[i].re].cost +=
			delta;
	cost += delta;
	return true;
}
void costflow() {
	do do memset(&vis[0], 0, sizeof(vis[0]) * vis.size());
	while (aug(src, INF)); while (modlabel());
}

namespace py = pybind11;

struct Edge{
    int u, v;
    double w;
    Edge(int u, int v, double w) : u(u), v(v), w(w) {}
};

py::list ZKW_algo(int N, int M, py::list inedge){
	std::vector<Edge> ev;
    double max_edge = -1e100;
    int en = 0;
    for (int ii = 0; ii < inedge.size(); ii ++ ){
        auto i = inedge[ii].cast<py::list>();
        double w = i[2].cast<double>();
        ev.push_back(Edge(i[0].cast<int>(), i[1].cast<int>(), w));
        max_edge = max_edge < w ? w : max_edge;
        en ++ ;
    }

    int MAXM = (en + N + M) * 2;
    int MAXN = N + M + 2;
    src = 0;
    des = N + M + 1;
    n = N + M + 2;

    init(MAXM, MAXN);
    
    for (int i = 0; i < N; i ++ )
        addedge(0, i + 1, 1, 0);
    for (int i = 0; i < M; i ++ )
        addedge(N + i + 1, N + M + 1, 1, 0);
    for (auto e : ev)
        addedge(e.u + 1, e.v + N + 1, 1, max_edge - e.w);
    costflow();

    py::list res;
    for (int i = 0; i < N; i ++ ){
        for (int j = head[i + 1]; j != -1; j = edge[j].next)
            if (!edge[j].cap){
                py::list l;
                l.append(i);
                l.append(edge[j].v - N - 1);
                res.append(l);
            }
    }
    return res;
}

PYBIND11_MODULE(ZKW, m){
    m.def("ZKW_algo", &ZKW_algo);
}

int main(){
    init(12, 5);
    addedge(0, 1, 2, 5);
    addedge(1, 3, 2, 2);
    addedge(0, 2, 2, 6);
    addedge(2, 3, 2, 0);
    addedge(3, 4, 3, 10.1);
    addedge(2, 1, 10, 0.1);
    src = 0;
    des = 4;
    n = 5;
    //for (auto i : head) std::cout << i << ' ';std::cout << std::endl;
    //for (auto i : edge) printf("(%d %d) ", i.next, i.v); std::cout << std::endl;
    costflow();
    std::cout << ans << ' ' << ans_flow << std::endl;
}

