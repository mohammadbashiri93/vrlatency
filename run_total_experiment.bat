:: call activate vrlatency3

:: for 120 Hz 1200 sample seems fine. "fine" means it covers the transition.
call measure_latency total --port COM11 --trials 300 --jitter --nsamples 1000 --stimsize 5 --stimdistance -1.5 --screen 1 --singlemode --output C:\Users\sirotalab\Desktop\Measurements