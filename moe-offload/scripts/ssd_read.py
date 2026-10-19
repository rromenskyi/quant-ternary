"""Random reads of expert-sized chunks from a big file, page cache bypassed
(F_NOCACHE): how fast experts can come off this Mac's SSD."""
import os, sys, time, random, fcntl, json
from concurrent.futures import ThreadPoolExecutor

F_NOCACHE = 48

def bench(path, chunk, threads, n=400):
    size = os.path.getsize(path)
    fds = []
    def worker(offsets):
        fd = os.open(path, os.O_RDONLY)
        fcntl.fcntl(fd, F_NOCACHE, 1)
        got = 0
        for off in offsets:
            got += len(os.pread(fd, chunk, off))
        os.close(fd)
        return got
    offs = [random.randrange(0, size - chunk) // 16384 * 16384 for _ in range(n)]
    parts = [offs[i::threads] for i in range(threads)]
    t = time.perf_counter()
    with ThreadPoolExecutor(threads) as ex:
        total = sum(ex.map(worker, parts))
    dt = time.perf_counter() - t
    return total / dt / 1e9

if __name__ == "__main__":
    path = sys.argv[1]
    for chunk_mb in [1.16, 3.35, 4.99]:
        for th in [1, 4, 8, 16]:
            gbps = bench(path, int(chunk_mb * 1e6), th, n=240)
            print(json.dumps({"chunk_MB": chunk_mb, "threads": th, "GB_per_s": round(gbps, 2)}))
