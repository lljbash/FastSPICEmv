#ifndef STOPWATCH_H
#define STOPWATCH_H

#include <omp.h>

class Stopwatch {
public:
    Stopwatch() : last_time_(omp_get_wtime()) {}

    double elapsed() {
        double this_time = omp_get_wtime();
        double delta_time = this_time - last_time_;
        last_time_ = this_time;
        return delta_time;
    }

private:
    double last_time_;
};

#endif /* ifndef STOPWATCH_H */
