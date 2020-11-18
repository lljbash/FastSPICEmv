#include <iostream>
#include <fstream>
#include <iterator>
#include <random>
#include <algorithm>
#include <vector>
#include <string>
#include <dlfcn.h>
#include "stopwatch.hpp"
#include "calc.h"

using namespace std;

#ifndef BASELINE
#undef BASELINE_TEST
#endif

//#define BASELINE

const string A_mat_filename = "A_mat.txt";
const string B_mat_filename = "B_mat.txt";
const string task_filename = "task.txt";
const char* baseline_lib = "./libnaive.so";
const char* my_lib = "./libcompetition.so";
const char* calc_taskA_func_name = "matrix_calc_taskA";
const char* calc_taskB_func_name = "matrix_calc_taskB";

random_device rd{};
//mt19937 gen{rd()};
mt19937 gen{16666};
normal_distribution<> randn;
uniform_real_distribution<> myrand{0., 1.};

auto randn_vector(int n) {
    vector<double> vec;
    generate_n(back_inserter(vec), n, [&](){ return randn(gen); });
    return vec;
}

struct CSR {
    int n;
    vector<int> offset;
    vector<int> indice;
    vector<double> value;
};

struct Task {
    CSR m;
    vector<double> v;
};

vector<Task> taskA_data;
vector<Task> taskB_data;

void read_CSR_structs_from_file(string filename, vector<Task>& task_data) {
    ifstream ifs(filename);
    int nn;
    ifs >> nn;
    task_data.resize(nn);
    string buf;
    for (int i = 0; i < nn; ++i) {
        ifs >> buf;
        ifs >> task_data[i].m.n;
        ifs >> buf;
        task_data[i].m.offset.resize(task_data[i].m.n + 1);
        for (int j = 0; j <= task_data[i].m.n; ++j) {
            ifs >> task_data[i].m.offset[j];
        }
        ifs >> buf;
        task_data[i].m.indice.resize(task_data[i].m.offset.back());
        for (int j = 0; j < task_data[i].m.offset.back(); ++j) {
            ifs >> task_data[i].m.indice[j];
        }
    }
};

void generate_taskA_data(vector<Task>& task_data) {
    for (auto& t : task_data) {
        t.m.value = randn_vector(t.m.offset.back());
        t.v = randn_vector(t.m.offset.back());
    }
}

void generate_taskB_data(vector<Task>& task_data) {
    for (auto& t : task_data) {
        t.m.value = randn_vector(t.m.offset.back() * 2);
        t.v = randn_vector(t.m.offset.back() * 2);
    }
}

using calc_taskA_func_type = decltype(matrix_calc_taskA);
using calc_taskB_func_type = decltype(matrix_calc_taskB);
calc_taskA_func_type* baseline_calc_taskA;
calc_taskB_func_type* baseline_calc_taskB;
void* baseline_so;

#ifdef BASELINE
__attribute__((constructor))
void load_baseline() {
    baseline_so = dlopen(baseline_lib, RTLD_NOW);
    baseline_calc_taskA = (calc_taskA_func_type*) dlsym(baseline_so, calc_taskA_func_name);
    baseline_calc_taskB = (calc_taskB_func_type*) dlsym(baseline_so, calc_taskB_func_name);
}

__attribute__((destructor))
void unload_baseline() {
    dlclose(baseline_so);
}
#endif

template <class func_type, class... T>
void call_func_from_lib(const char* lib_name, const char* func_name, double& runtime, T&&... args) {
    void* myso = dlopen(lib_name, RTLD_NOW);
    func_type* func = (func_type*) dlsym(myso, func_name);
    Stopwatch timer;
    func(forward<T&&>(args)...);
    runtime += timer.elapsed();
    dlclose(myso);
}

template <class func_type, class... T>
void call_func(func_type* func, double& runtime, T&&... args) {
    Stopwatch timer;
    timer.elapsed();
    func(forward<T&&>(args)...);
    runtime += timer.elapsed();
}

int64_t total_mem_acc = 0;
int64_t total_float_op = 0;

double baseline_runtimeA = 0;
double my_runtimeA = 0;
double baseline_runtimeB = 0;
double my_runtimeB = 0;

void update_err(double baseline, double my, double& max_err) {
    double err = fabs(baseline - my);
    if (err > max_err) {
        max_err = err;
    }
}

void test_A(istream& is) {
    vector<vector<int>> row_array;
#ifdef BASELINE
    vector<TaskMatrixInfoA*> baseline_info;
    vector<vector<double>> baseline_Id;
#endif
    vector<TaskMatrixInfoA*> my_info;
    vector<vector<double>> my_Id;
    for (auto& t : taskA_data) {
        int n;
        is >> n;
        row_array.emplace_back(n);
        for (int& r : row_array.back()) {
            is >> r;
        }
#ifdef BASELINE
        baseline_Id.emplace_back(t.m.n);
#endif
        my_Id.emplace_back(t.m.n);
        int nnz = t.m.offset.back();
        total_mem_acc += sizeof(int[t.m.n + nnz]) + sizeof(double[t.m.n * 2 + nnz]);
        total_float_op += nnz * 2;
    }
    int nn = (int) taskA_data.size();
    for (int i = 0; i < nn; ++i) {
#ifdef BASELINE
        baseline_info.push_back(new TaskMatrixInfoA {
            .rowArray = row_array[i].data(),
            .rowOffset = taskA_data[i].m.offset.data(),
            .rowArraySize = (int) row_array[i].size(),
            .columnIndice = taskA_data[i].m.indice.data(),
            .S = taskA_data[i].v.data(),
            .valueNormalMatrix = taskA_data[i].m.value.data(),
            .Id = baseline_Id[i].data()
        });
#endif
        my_info.push_back(new TaskMatrixInfoA {
            .rowArray = row_array[i].data(),
            .rowOffset = taskA_data[i].m.offset.data(),
            .rowArraySize = (int) row_array[i].size(),
            .columnIndice = taskA_data[i].m.indice.data(),
            .S = taskA_data[i].v.data(),
            .valueNormalMatrix = taskA_data[i].m.value.data(),
            .Id = my_Id[i].data()
        });
    }
    //call_func_from_lib<calc_taskA_func_type>(baseline_lib, calc_taskA_func_name, baseline_runtimeA, baseline_info.data(), nn);
    //call_func_from_lib<calc_taskA_func_type>(my_lib, calc_taskA_func_name, my_runtime, my_info.data(), nn);
#ifdef BASELINE
    double max_err = 0;
    call_func(baseline_calc_taskA, baseline_runtimeA, baseline_info.data(), nn);
#endif
    call_func(matrix_calc_taskA, my_runtimeA, my_info.data(), nn);
    for (int i = 0; i < nn; ++i) {
#ifdef BASELINE_TEST
        for (int j = 0; j < taskA_data[i].m.n; ++j) {
            update_err(baseline_info[i]->Id[j], my_info[i]->Id[j], max_err);
        }
        delete baseline_info[i];
#endif
        delete my_info[i];
    }
#ifdef BASELINE_TEST
    cerr << "TaskA max_err = " << scientific << max_err << endl;
#endif
}

void test_B(istream& is) {
    vector<vector<int>> row_array;
#ifdef BASELINE
    vector<TaskMatrixInfoB*> baseline_info;
    vector<vector<double>> baseline_A;
    vector<vector<double>> baseline_R;
    vector<vector<double>> baseline_H;
    vector<vector<double>> baseline_D;
    vector<vector<double>> baseline_IC;
    vector<vector<double>> baseline_IG;
#endif
    vector<TaskMatrixInfoB*> my_info;
    vector<vector<double>> my_A;
    vector<vector<double>> my_R;
    vector<vector<double>> my_H;
    vector<vector<double>> my_D;
    vector<vector<double>> my_IC;
    vector<vector<double>> my_IG;
    for (auto& t : taskB_data) {
        int n;
        is >> n;
        row_array.emplace_back(n);
        for (int& r : row_array.back()) {
            is >> r;
        }
        auto tmp = randn_vector(t.m.n * 2);
#ifdef BASELINE
        baseline_A.emplace_back(t.m.offset.back());
        baseline_R.emplace_back(t.m.n);
        baseline_H.emplace_back(t.m.n);
        baseline_D.push_back(tmp);
        baseline_IC.emplace_back(t.m.n);
        baseline_IG.emplace_back(t.m.n);
#endif
        my_A.emplace_back(t.m.offset.back());
        my_R.emplace_back(t.m.n);
        my_H.emplace_back(t.m.n);
        my_D.push_back(tmp);
        my_IC.emplace_back(t.m.n);
        my_IG.emplace_back(t.m.n);
        int nnz = t.m.offset.back();
        total_mem_acc += sizeof(int[t.m.n + nnz]) + sizeof(double[t.m.n * 7 + nnz * 3]);
        total_float_op += nnz * 5;
    }
    int nn = (int) taskB_data.size();
    for (int i = 0; i < nn; ++i) {
        double alpha = myrand(gen);
#ifdef BASELINE
        baseline_info.push_back(new TaskMatrixInfoB {
            .valueSpiceMatrix = taskB_data[i].m.value.data(),
            .rowOffset = taskB_data[i].m.offset.data(),
            .columnIndice = taskB_data[i].m.indice.data(),
            .A = baseline_A[i].data(),
            .S = taskB_data[i].v.data(),
            .R = baseline_R[i].data(),
            .H = baseline_H[i].data(),
            .D = baseline_D[i].data(),
            .IC = baseline_IC[i].data(),
            .IG = baseline_IG[i].data(),
            .alpha = alpha,
            .rowArray = row_array[i].data(),
            .rowArraySize = (int) row_array[i].size(),
            .hdl = nullptr
        });
#endif
        my_info.push_back(new TaskMatrixInfoB {
            .valueSpiceMatrix = taskB_data[i].m.value.data(),
            .rowOffset = taskB_data[i].m.offset.data(),
            .columnIndice = taskB_data[i].m.indice.data(),
            .A = my_A[i].data(),
            .S = taskB_data[i].v.data(),
            .R = my_R[i].data(),
            .H = my_H[i].data(),
            .D = my_D[i].data(),
            .IC = my_IC[i].data(),
            .IG = my_IG[i].data(),
            .alpha = alpha,
            .rowArray = row_array[i].data(),
            .rowArraySize = (int) row_array[i].size(),
            .hdl = nullptr
        });
    }
    //call_func_from_lib<calc_taskB_func_type>(baseline_lib, calc_taskB_func_name, baseline_runtimeB, baseline_info.data(), nn);
    //call_func_from_lib<calc_taskB_func_type>(my_lib, calc_taskB_func_name, my_runtime, my_info.data(), nn);
#ifdef BASELINE
    double max_err = 0;
    call_func(baseline_calc_taskB, baseline_runtimeB, baseline_info.data(), nn);
#endif
    call_func(matrix_calc_taskB, my_runtimeB, my_info.data(), nn);
    for (int i = 0; i < nn; ++i) {
#ifdef BASELINE_TEST
        for (int j = 0; j < taskB_data[i].m.n; ++j) {
            update_err(baseline_info[i]->R[j], my_info[i]->R[j], max_err);
            update_err(baseline_info[i]->H[j], my_info[i]->H[j], max_err);
            update_err(baseline_info[i]->IC[j], my_info[i]->IC[j], max_err);
            update_err(baseline_info[i]->IG[j], my_info[i]->IG[j], max_err);
        }
        for (int j = 0; j < taskB_data[i].m.offset.back(); ++j) {
            update_err(baseline_info[i]->A[j], my_info[i]->A[j], max_err);
        }
        delete baseline_info[i];
#endif
        delete my_info[i];
    }
#ifdef BASELINE_TEST
    cerr << "TaskB max_err = " << scientific << max_err << endl;
#endif
}

void test_all_cases() {
    baseline_runtimeA = 0;
    baseline_runtimeB = 0;
    my_runtimeA = 0;
    my_runtimeB = 0;
    ifstream ifs(task_filename);
    while (true) {
        string buf;
        ifs >> buf;
        if (buf == "A") {
            test_A(ifs);
        }
        else if (buf == "B") {
            test_B(ifs);
        }
        else {
            break;
        }
    }
#ifdef BASELINE
    cerr << "baseline time: " << fixed << baseline_runtimeA << "s " << baseline_runtimeB << "s " << baseline_runtimeA + baseline_runtimeB << "s" << endl;
#endif
    cerr << "my time:       " << fixed << my_runtimeA << "s " << my_runtimeB << "s " << my_runtimeA + my_runtimeB << "s" << endl;
    //cerr << "total_mem_acc: " << total_mem_acc << endl;
    //cerr << "total_float_op: " << total_float_op << endl;
}

int main() {
    read_CSR_structs_from_file(A_mat_filename, taskA_data);
    read_CSR_structs_from_file(B_mat_filename, taskB_data);
    generate_taskA_data(taskA_data);
    generate_taskB_data(taskB_data);
    for (int i = 0; i < 10; ++i) {
        test_all_cases();
    }
    return 0;
}
