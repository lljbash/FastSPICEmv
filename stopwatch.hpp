#ifndef STOPWATCH_H
#define STOPWATCH_H

#include <tbb/tick_count.h>

class Stopwatch {
public:
    Stopwatch() : last_time_(tbb::tick_count::now()) {}

    double elapsed() {
        auto this_time = tbb::tick_count::now();
        auto delta_time = this_time - last_time_;
        last_time_ = this_time;
        return delta_time.seconds();
    }

private:
    tbb::tick_count last_time_;
};

#endif /* ifndef STOPWATCH_H */
