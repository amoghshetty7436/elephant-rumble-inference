---
description: How the detector's inference can be split across CPU cores, GPUs and several machines, and why reading the audio (communication) can be hidden behind AVES (compute) on CPUs but not on a fast GPU. Read for any question about scaling out, parallel workers or hardware detection.
---

# Splitting the detector's work across CPUs, GPUs and machines

All the numbers here are from the lab PC (24 logical CPU cores, RTX 4060), measured with `tools/cpu_speed.py` and `tools/detect.py` on 27 October 2026.

## 1. Where the time goes for one 24-hour recording

| Step | What it is | Lab PC |
|---|---|---|
| Read + resample | Load 1.3 GB of audio and turn it into 500 samples a second | 6 s from the local disk, 19 s from the L: drive (one CPU core) |
| AVES | Turn the audio into fingerprints | GPU: 4 s (half precision). CPU: 79 s on all 24 cores |
| Classifier + alarm rule | Probabilities, alarms, Raven table | 0.1 s |

AVES is the **compute**. Reading the file is the **communication**: moving the data to where the compute happens. Writing the Raven table is also communication, but it is a few kilobytes, so it doesn't matter.

## 2. Two ways to use many cores

**Inside one recording (threads).** PyTorch splits AVES's big matrix multiplications across cores. This is automatic: `--threads 24` just tells PyTorch how many to use. It scales poorly, because every step has to wait for the slowest core and the cores share memory bandwidth:

| Threads on one recording | Time per 24 h | Speed-up |
|---|---|---|
| 1 | 8 m 46 s | 1× |
| 4 | 3 m 43 s | 2.4× |
| 24 | 1 m 19 s | 6.7× (not 24×) |

**Across recordings (workers).** Each recording is completely independent of the others, so separate processes can each take a whole recording. Nothing has to be shared while they work. This is called *embarrassingly parallel*, and it is the easy case. `--workers 6 --threads 4` runs 6 recordings at once, each with 4 cores. For a folder of hundreds of days, this gives more total throughput than one recording on 24 threads.

The rule of thumb: **use few threads per recording and many recordings at once.** Use all threads on one recording only when you need one file back fast.

## 3. Does the model need to know the hardware?

No. The model is just a file of weights. The same weights run on any CPU or GPU. What changes is the **launcher**, the script that decides how to split the work. It can read the hardware at start-up:

- CPU cores: `os.cpu_count()` gives logical cores (24 on the lab PC). Physical cores are fewer when the CPU has hyper-threading. `psutil.cpu_count(logical=False)` gives the physical count.
- GPUs: `torch.cuda.device_count()`, and for each one `torch.cuda.get_device_name(i)` and its memory.
- Memory: each CPU worker holds AVES (about 0.4 GB), a day of audio (about 0.2 GB per day read ahead) and working memory, so roughly 1.5–2 GB per worker. The number of workers must fit in RAM.

From that it picks a plan, so the same command works on a laptop, the lab PC or a server. At the moment `detect.py` has `--device auto` (GPU if there is one), and you set `--workers` and `--threads` yourself. Making it choose those automatically is a small step (see section 7).

## 4. GPU cores are different

A GPU has thousands of small cores (the RTX 4060 has 3,072), but you never hand work to them one by one. You give the GPU one big operation, such as "AVES on this hour of audio", and its hardware spreads it over all its cores by itself. So the unit we schedule is **a whole GPU**: one process per GPU card, each taking whole recordings. With 2 GPUs you run 2 GPU workers.

## 5. GPU and CPU at the same time?

Possible: put all recordings in one shared queue, and let a GPU worker and some CPU workers each take the next file when they are free. This is dynamic load balancing: the faster worker simply takes more files.

On the lab PC it isn't worth it:
- The GPU does a day in about 4 s. All 24 CPU cores together take about 79 s. So the CPU would add only about 5% more throughput.
- The GPU needs those CPU cores to read and resample the audio fast enough to keep it busy. If the CPU cores also run AVES, the readers slow down and the GPU waits.

It pays off only where the CPU is a large share of the total power, for example a server with many cores and a small GPU. On ELP's CPU-only servers there is no choice to make: everything runs on the CPU.

## 6. Scaling up and scaling out

- **Scaling up:** a bigger machine (more cores, a faster GPU). There is a limit, and it gets expensive.
- **Scaling out:** more machines, each doing its own share. For us this is simple. Give machine 1 the first 1/N of the recordings, machine 2 the next 1/N, and so on. Each one writes its own Raven tables. The machines never talk to each other while they work, so adding machines adds speed almost perfectly. ELP's plan, 4,500 recording-days on about 30 CPU cores, is this kind of scaling out.

## 7. Can the communication be hidden behind the compute?

"Hiding" means reading the next recording while the current one is in AVES, which is what `--readers` does. It works fully when reading is faster than computing. Then the compute never waits.

**On the CPU: yes, easily.** AVES takes about 32 core-minutes per day. Reading takes 6–19 s. With 6 workers × 4 cores, each worker spends minutes on AVES per day and seconds reading. The reading is hidden almost completely.

**On a fast GPU: only partly.** AVES takes 4 s per day, but reading takes 6 s from the local disk and 19 s from the L: drive. Reading is now slower than computing, so the GPU waits. Reading several files at once (`--readers 3`) helps, which is why the run did a day every 4 s. But a slow disk or network sets the limit, however fast the GPU is.

**At ELP's scale, the network matters most.** 4,500 days × 1.3 GB is about 5.8 TB of audio.

| Link | Time just to move 5.8 TB |
|---|---|
| 1 Gbit/s network (about 110 MB/s) | about 15 hours |
| 10 Gbit/s network | about 1.5 hours |
| Local SSD (about 1–3 GB/s) | under 2 hours |

- On 30 CPU cores, the compute takes about 3.3 days, so even the 15 hours on a 1 Gbit/s network is easy to hide. **Compute-bound.**
- On one RTX 4060, the compute takes about 5 hours, so a 1 Gbit/s network (15 hours) would be the limit. **Communication-bound.** Here you would copy the audio to fast local disks first, or spread the files over several machines, each with its own disk.

Moving data from the CPU to the GPU (over PCIe) is not an issue. An hour of 500 Hz audio is about 7 MB, which takes about a millisecond.

## 8. What `detect.py` does today, and what could come next

Today:
- `--device auto|cpu|cuda`, `--workers N`, `--threads T` and `--readers N` (read ahead in the background).
- Workers get a fixed share of the files: file 1, 7, 13… to worker 1 when there are 6 workers. If some files are much longer than others, one worker can finish late.

Possible next steps, if the CPU results call for them:
- `--workers auto`: pick workers and threads from the cores and RAM, for example 4 threads per worker.
- A shared queue instead of fixed shares, so a worker that finishes early takes the next file.
- `--shard i/N`: process only the i-th of N parts of a folder, so N machines can split a big job with one command each.
- A run on the GPU and CPU from one queue, if ELP gets machines where that pays.

## 9. Many machines with different hardware: one launcher, a shared queue

You don't need a launcher for each combination of machines, and you don't need to work out the best split in advance. The standard pattern is a **work queue (pull, not push)**:

1. All recordings go in one shared to-do list.
2. Every machine runs the same command. At start-up the launcher sizes itself to that machine: one worker per GPU, plus CPU workers of about 4 threads each, as many as fit in its cores and RAM.
3. Each worker takes the next recording from the list when it is free, processes it and writes its Raven table. Then it takes another.

A fast machine finishes recordings sooner, so it simply takes more of them. A slow one takes fewer. The split balances itself, and all machines finish within about one recording's time of each other. That is close to the best possible split, with no planning. If a machine crashes, its unfinished recording goes back on the list.

**Where the queue lives.** For ELP the simplest is a shared network folder. A worker claims a recording by creating a small "claimed" file next to its output; Windows guarantees only one worker can create it. No extra software is needed. Bigger set-ups use a scheduler instead: a cluster's job system (for example SLURM job arrays) or a Python framework such as Ray or Dask.

**The one thing to tune per machine** is threads per worker. The launcher can find it by itself: time AVES on a minute of audio with 2, 4 and 8 threads at start-up, and keep the fastest per core.

**Why we don't split the model itself across machines.** You can split one model's layers across devices (pipeline or model parallelism), but that is for models too big for one device. AVES (about 95 million weights, 0.4 GB) fits anywhere, and passing its intermediate results between machines would cost far more than computing them. Splitting by recording (data parallelism) has no traffic between workers at all.

**What does need watching** is the shared storage. All machines read their audio from the same place, so that file server's network link carries all 5.8 TB (section 7).
