#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace Plot {
  class ThreadPool {
public:
    ThreadPool(size_t threads) {
      for (size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this]() {
          while (true) {
            std::function<void()> task;
            {
              std::unique_lock<std::mutex> lock(queueMutex);
              condition.wait(lock, [this]() { return !tasks.empty() || stop; });
              if (stop && tasks.empty())
                return;
              task = std::move(tasks.front());
              tasks.pop();
            }
            task();
          }
        });
      }
    }

    ~ThreadPool() {
      {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
      }
      condition.notify_all();
      for (std::thread &worker: workers)
        worker.join();
    }

    void enqueue(std::function<void()> task) {
      {
        std::unique_lock<std::mutex> lock(queueMutex);
        tasks.push(std::move(task));
      }
      condition.notify_one();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop = false;
  };
} // namespace Plot
