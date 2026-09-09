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

/**
 * Accumulating timers for the stages *inside* the agglomeration loop, so the
 * CPU kernels can be compared against the GPU pipeline stage by stage.  The
 * loop runs N'-3 times, so a per-iteration PhaseTimer would emit tens of
 * thousands of lines; these buckets add up instead and are reported once.
 *
 * The stage names deliberately mirror the GPU kernels of Algorithm 1:
 *
 *   K2  optimal-pair search   getMinimumEntry() / getRowMinima()
 *   K3  variance weight       chooseLambda()          (BIONJ only)
 *   K4  matrix update         the D/V/R update loop, including the row total
 *   K5  hole-filling          removeRowAndColumn()
 *
 * RapidNJ's pruned search needs bookkeeping that has no GPU counterpart (the
 * sorted S and I matrices, periodic purging, cluster-total maintenance), so it
 * gets a bucket of its own rather than being folded into one of the above.
 *
 * Every StageTimer scope is entered on the master thread, outside any OpenMP
 * parallel region, so plain doubles need no synchronisation.  Scopes must not
 * nest within the same stage; where a stage brackets a call that is itself
 * instrumented, use two adjacent scopes instead.
 */
enum LoopStage {
    STAGE_PAIR_SEARCH = 0,
    STAGE_LAMBDA,
    STAGE_UPDATE,
    STAGE_COMPACT,
    STAGE_RNJ_UPKEEP,
    STAGE_COUNT
};

inline double* loopStageSeconds() {
    static double seconds[STAGE_COUNT] = {0};
    return seconds;
}

inline long long* loopStageCalls() {
    static long long calls[STAGE_COUNT] = {0};
    return calls;
}

class StageTimer {
public:
    explicit StageTimer(LoopStage stage)
        : stage_(stage), on_(phaseTimingEnabled())
        , start_(on_ ? getRealTime() : 0.0) {}
    ~StageTimer() {
        if (on_) {
            loopStageSeconds()[stage_] += getRealTime() - start_;
            ++loopStageCalls()[stage_];
        }
    }
private:
    LoopStage stage_;
    bool      on_;
    double    start_;
};

inline void reportLoopStages() {
    if (!phaseTimingEnabled()) {
        return;
    }
    static const char* names[STAGE_COUNT] = {
        "K2 optimal-pair search", "K3 variance weight",
        "K4 matrix update", "K5 hole-filling",
        "RapidNJ S/I upkeep"
    };
    const double*    secs  = loopStageSeconds();
    const long long* calls = loopStageCalls();
    double total = 0.0;
    for (int s = 0; s < STAGE_COUNT; ++s) {
        total += secs[s];
    }
    std::streamsize old = std::cout.precision(6);
    for (int s = 0; s < STAGE_COUNT; ++s) {
        if (calls[s] == 0) {
            continue;
        }
        std::cout << "[stage] " << names[s] << " : " << secs[s]
                  << " sec wall, " << calls[s] << " calls, "
                  << (total > 0 ? 100.0 * secs[s] / total : 0.0)
                  << " pct of loop" << std::endl;
    }
    std::cout << "[stage] total instrumented loop : " << total
              << " sec wall" << std::endl;
    std::cout.precision(old);
}

#endif /* PHASETIMER_H */
