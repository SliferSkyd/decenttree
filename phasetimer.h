#ifndef PHASETIMER_H
#define PHASETIMER_H

/**
 * Optional per-phase wall/CPU timing, enabled with -phase-time.
 *
 * decentTree normally reports a single "Computing <ALGO> tree took ..." line,
 * which lumps the alignment load, the distance-matrix build, duplicate
 * clustering and the algorithm itself together.  For BIONJ-GPU that hides
 * where the time actually goes (on large inputs the distance matrix dominates,
 * and the GPU joins are a small fraction).  These timers break it down.
 *
 * Output lines are prefixed "[phase] " so they are trivial to grep, and are
 * only emitted when -phase-time is passed, so default output is unchanged.
 *
 * The enable flag lives in an inline function's static, so a single instance
 * is shared across translation units (including the hipcc-compiled kernel TU)
 * without needing C++17 inline variables.
 */

#include <iostream>
#include <string>
#include <utils/timeutil.h>

inline bool& phaseTimingEnabled() {
    static bool enabled = false;
    return enabled;
}

class PhaseTimer {
public:
    explicit PhaseTimer(const std::string& name)
        : name_(name), wall_(getRealTime()), cpu_(getCPUTime())
        , reported_(false) {}

    /** Report elapsed time (idempotent; the destructor calls it too). */
    void report() {
        if (!phaseTimingEnabled() || reported_) {
            return;
        }
        reported_ = true;
        std::streamsize old = std::cout.precision(6);
        std::cout << "[phase] " << name_
                  << " : " << (getRealTime() - wall_) << " sec wall, "
                  << (getCPUTime()  - cpu_)  << " sec cpu" << std::endl;
        std::cout.precision(old);
    }

    /** Elapsed wall-clock seconds so far, without reporting. */
    double elapsed() const { return getRealTime() - wall_; }

    ~PhaseTimer() { report(); }

private:
    std::string name_;
    double      wall_;
    double      cpu_;
    bool        reported_;
};

#endif /* PHASETIMER_H */
