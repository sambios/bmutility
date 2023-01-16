//
// Created by yuan on 2/26/21.
//

#ifndef BMUTILITY_TIMER_H
#define BMUTILITY_TIMER_H

#include <iostream>
#include <memory>
#include <functional>
#include <chrono>

namespace bm {

    uint64_t gettime_sec();
    uint64_t gettime_msec();
    uint64_t gettime_usec();
    void msleep(int msec);
    void usleep(int usec);
    std::string timeToString(time_t seconds);

    class TimerQueue {
    public:
        static std::shared_ptr<TimerQueue> create();
        virtual ~TimerQueue() {
            std::cout << "TimerQueue dtor" << std::endl;
        };

        virtual int create_timer(uint32_t delay_msec, std::function<void()> func, int repeat, uint64_t *p_timer_id) = 0;
        virtual int delete_timer(uint64_t timer_id) = 0;
        virtual size_t count() = 0;
        virtual int run_loop() = 0;
        virtual int stop() = 0;
    };

    using TimerQueuePtr = std::shared_ptr<TimerQueue>;

    class StatTool {
    public:
        static std::shared_ptr<StatTool> create(int range=5);
        virtual ~StatTool(){};

        virtual void update(uint64_t currentStatis) = 0;
        virtual void reset() = 0;
        virtual double getkbps() = 0;
        virtual double getSpeed() = 0;
    };

    using StatToolPtr = std::shared_ptr<StatTool>;

    class BMPerf {
        int64_t start_us_;
        std::string tag_;
        int threshold_{50};
    public:
        BMPerf() {}
        BMPerf(const std::string &name, int threshold = 0) {
            begin(name, threshold);
        }
        ~BMPerf() {}

        void begin(const std::string &name, int threshold = 0) {
            tag_ = name;
            auto n = std::chrono::steady_clock::now();
            start_us_ = n.time_since_epoch().count();
            threshold_ = threshold;
        }

        void end() {
            auto n = std::chrono::steady_clock::now().time_since_epoch().count();
            auto delta = (n - start_us_) / 1000;
            if (delta < threshold_ * 1000) {
                //printf("%s used:%d us\n", tag_.c_str(), delta);
            } else {
                printf("WARN:%s used:%d ms > %d ms\n", tag_.c_str(), (int)delta/1000, (int)threshold_);
            }
        }
    };
}


#endif //INFERENCE_FRAMEWORK_BMUTILITY_TIMER_H
