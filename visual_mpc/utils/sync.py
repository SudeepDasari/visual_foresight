from multiprocessing import Value, Lock


class SyncCounter:
    def __init__(self, base_value=0):
        self._lock = Lock()
        self._value = Value('i', base_value)

    @property
    def ret_increment(self):
        with self._lock:
            ret_val = self._value.value
            self._value.value += 1
        return ret_val

    @property
    def value(self):
        with self._lock:
            ret_val = self._value.value
        return ret_val


class ManagedSyncCounter(SyncCounter):
    def __init__(self, manager, base_value=0):
        self._lock, self._value = manager.Lock(), manager.Value('i', base_value)
