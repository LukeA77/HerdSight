# S6 usability test: plan

**Target (S6):** a non-technical person can tell within 10 seconds whether any camp needs attention, and can act on an
alert without help. Five testers; every result is reported, pass or fail, per tester.

## Testers
Five adults who do not work in software or data, ideally people who keep or work with livestock. Nobody who has seen
the dashboard before. Each session takes about 15 minutes.

## Set-up
- The made-up farm: `python -m herdsight.dashboard.server --db s6.sqlite --mock`, opened on a phone (or a laptop
  browser narrowed to phone width if no phone can reach the laptop). Start a fresh database for every tester so
  earlier taps do not carry over.
- A timer, and the recording sheet below. The facilitator only reads the script and does not point or explain.

## Script (read out as written)
1. "This is an app that shows the result of a drone counting cattle on a farm. Please think aloud as you use it."
2. **Task 1 (the 10-second test).** Start the timer when the farm page appears. "Does any part of the farm need your
   attention? Which one is the most urgent?" Stop the timer when the tester names a camp.
3. **Task 2.** "Find out what the app thinks has happened in that camp, and how sure it is."
4. **Task 3.** "Suppose you went out and found the animals. Tell the app." (Expected: open the alert and press
   "I found them".)
5. **Task 4.** "One camp says it may have animals that moved. Which camps are involved?"
6. **Task 5.** "Turn off your phone's signal (or the laptop's Wi-Fi), reopen the app and find the number of cows in
   Top camp." (Checks the weak-signal behaviour.)
7. Two questions at the end: "Was anything confusing?" and "Would you trust this to tell you about missing animals?
   Why or why not?"

## Pass rule (fixed before any session)
- Task 1: the tester names Bush camp (the red one) as most urgent within 10 seconds, without help.
- Task 3: the tester records "I found them" on the Bush camp alert without help, within 60 seconds.
- A tester passes S6 if both hold. S6 is reported as the number of testers who pass (for example "4 of 5"), with
  every tester's times and comments. Tasks 2, 4 and 5 and the two questions are reported as findings, not pass or
  fail.

## Recording sheet (one per tester)
| Item | Result |
|---|---|
| Tester (initials only), date | |
| Task 1: camp named, seconds | |
| Task 2: what they said (short) | |
| Task 3: done without help? seconds | |
| Task 4: camps named | |
| Task 5: worked without signal? | |
| Confusing parts | |
| Trust (their words) | |
| S6 pass? | |

Store the sheets privately; only initials, times and anonymised comments go into the results file.
