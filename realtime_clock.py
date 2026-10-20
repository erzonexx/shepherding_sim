"""Fixed-rate physical clock. Waiting is for the next tick, never for a planner."""
import time


class RealtimeClock:
    def __init__(self, hz=30.0):
        if hz<=0:
            raise ValueError('Control frequency must be positive')
        self.period=1.0/hz
        self.start=time.perf_counter()
        self.lateness=[]
        self.step_seconds=[]

    def tick(self, frame):
        deadline=self.start+frame*self.period
        remaining=deadline-time.perf_counter()
        if remaining>0:
            time.sleep(remaining)
        now=time.perf_counter()
        self.lateness.append(max(0.,now-deadline))
        return now

    def record_step(self, started):
        self.step_seconds.append(time.perf_counter()-started)
