# Captions for the README demo clips

`scripts/make_demo_gifs.py` reads the captions for both demo clips from this file. To change a caption, edit the text under its heading and run the script again.

- Leave the `###` heading lines as they are; the code before the first space is how the script finds each caption.
- Words in curly brackets, like `{photos}`, are numbers filled in from the results. They can be moved or removed, but the word inside the brackets must stay the same.
- A time in a heading, like `(3.0 s)`, is how long that caption stays on screen.

## Clip 1: the survey (simulation)

### 1-01 Title card, big heading (2.0 s)
The drone surveys the farm

### 1-02 Title card, small line above the heading
Step 1

### 1-03 Title card, small line below the heading
A simulated drone and farm.

### 1-04 Key, before the drone moves (2.0 s): red dot, then this word
drone

### 1-04b Key, second item: white dot, then this word
cow

### 1-05 While flying, first part
The drone flies back and forth over the herd of cows.

### 1-06 While flying, middle part
It takes overlapping photos every few seconds.

### 1-07 While flying, last part
Until the entire area is captured.

### 1-08 End of the flight (3.5 s). {photos} = number of photos
This example has a total of {photos} photos. In the next step, the system counts all the cows it flew over.

## Clip 2: counting the cows (real photos)

### 2-01 Title card, big heading (3.0 s)
Counting the cows in the photos

### 2-02 Title card, small line above the heading
Step 2

### 2-03 Title card, small line below the heading
There will be two photos taken a few seconds apart

### 2-04 Map: only the blue frame (photo 1) shown (3.2 s)
The blue frame is the first photo taken.

### 2-05 Map: pink frame (photo 2) and the arrow appear (4.2 s)
The second pink frame is taken a few seconds later

### 2-06 Map: the shared ground lights up (4.5 s)
The two photos overlap. Any cows in the light area are in both photos.

### 2-07 Map: white box around the herd (4.0 s)
The white box marks where the herd is.

### 2-08 The two photos side by side, no boxes yet (5.2 s)
The darker part on the left is only in the first photo and the lighter part repeats in photo 2.

### 2-09 Green boxes on every cow (5.0 s). {photo1_cows}, {photo2_cows} = cows found in each photo
The drone puts a green box on every cow it finds: {photo1_cows} in photo 1 and {photo2_cows} in photo 2.

### 2-10 First two example lines appear (1.6 s each)
Then a yellow line links to that same cow in photo 2.

### 2-11 Last two example lines appear (3.2 s on the last one)
This represents cows that have been seen twice, once in each photo.

### 2-12 All the lines, matched cows turn yellow (5.2 s). {lines} = number of lines
The drone finds {lines} cows that are already counted. Each with a line, and a yellow box.

### 2-13 Lines fade, yellow boxes stay (5.0 s)
The cows in yellow boxes are not recounted.

### 2-14 Photo 1 cows with no partner circled in light blue (6.2 s). {photo1_alone} = number circled
The {photo1_alone} circled cows in photo 1 stay green. They were either missed or not in photo 2.

### 2-15 New cows in photo 2 circled in white (5.5 s). {new} = number circled
The {new} circled cows in photo 2 stay green: they were not in photo 1.

### 2-16 The counter goes up (4.5 s). {photo1_cows}, {new}, {count}
Photo 1 = {photo1_cows} cows, Photo 2 = {new} cows, total = {count} cows.

### 2-17 End card, big line (3.0 s). {count} = the count
HerdSight counted {count} different cows

### 2-18 End card, second line. {hand} = the hand count of the same two photos
Correct Headcount = {hand}

### 2-19 End card, small line added a moment later (5.0 s). {difference} = how far off it was
In this example {difference} additional cows counted.

### 2-19b The same line if it ever counts too few. {difference}
In this example {difference} cows were missed.

### 2-19c The same line if it ever gets the same number
In this example the count was exactly right.

## Small labels inside the pictures

### L-01 Label on the survey map
Simulation

### L-02 Counter on the survey map
Photos taken

### L-03 Label on the left photo
Photo 1

### L-03b Label on the right photo
Photo 2

### L-04 Arrow label on the map in clip 2
drone flying this way

### L-05 Counter in the caption strip of clip 2
Cows counted
