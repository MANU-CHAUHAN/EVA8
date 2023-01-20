# Session 5: Coding Drill Down

TODO:

1. achieve 99.4% (this must be consistently shown in your last 4 epochs, and not a one-time achievement)

2. to use Less than or equal to 15 Epochs

3. to use Less than 8000 Parameters (additional points for doing this in less than 8000 pts)

4. to meet this additional specification
   a) use Dropout with 0.05 value in all but one layer
   b) use rotation as an augmentation strategy (± 6.9 degrees)

---

The idea for the coding drill down was to practice how to approach a CNN architecture for a given problem.

For this exercise the first code target was to have a decently sized model with decent learning capabilities, without being fancy with additional optimizations or schedulers or augmentatin strategies. (1st step for any problem: keep it simple silly)

Target for second code was to have reduced gap between train and test accuracies as well as push model towards 99.4% target, with help of Dropout and Image Augmentation to reduce over-fitting as well as ease out the variances with augmentation and help network learn better.

Finally, for last code target was to have test accuracy consistently be above 99.40% (at least last 4 epochs), push the learning capacity of the model further while ensuring the gap between the two accuracies stays close.

The analysis for each code part is mentioned at the end of corresponding notebooks. The format for each notebook is same, `EVA 6 Code #`. Please go through the notebooks for target, results and analysis.
