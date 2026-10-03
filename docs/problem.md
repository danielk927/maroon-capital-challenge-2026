# Challenge problem (Maroon Capital, 2026)

The problem statement as given to applicants, with PDF line breaks and page numbers removed.

## Problem Statement

You are competing against other applicants.
Each person has a limited number of points to distribute across several battlefields.
For each battlefield, whoever allocates the most points wins that battlefield.
Your goal is to outperform the other applicants on as many battlefields as possible, forcing you to decide where to focus your resources and where you are willing to give ground.

There are 10 battlefields, worth 1, 2, 3 . . . 10 points respectively if they are won.
You have 100 units of troops in total to divide across the 10 battlefields.
You must devote strictly more troops than your opponent to win a battlefield.
In the case of a tie, no points are granted for that battlefield.
Your base score is the sum of points from battlefields won.
Additionally the following rules are imposed in order:

- If you win two adjacent battlefields, each score will be multiplied by 1.5.
  This will not "stack up" so if you have 1, 2, and 3, with 1 and 3 adjacent to 2, 2 will only be given the modifier once.
- If your base score is a multiple of 5, your score is further multiplied by 1.4.
- If you win the battlefield with value 6, your score is further multiplied by 1.1.

Your total score is the score after applying all three rules above.
You win against another applicant by having a higher total score.
As an example, if you win battlefields 3, 5, 6, and 7, your score will be calculated as follows:

- base score = 3 + 5 + 6 + 7 = 21
- Rule 1: score1 = 3 + 5 * 1.5 + 6 * 1.5 + 7 * 1.5 = 30
- Rule 2 - since base score equals 21, score is unchanged (score2 = score1 = 30)
- total score = score3 = 1.1 * score2 = 1.1 * 30 = 33

We will run everyone's submission against each other.
We will evaluate based on number of wins and total points.

## Submission

Submit your response in the Google Form by Tuesday, October 6th at midnight.
Your submission must include 10 non-negative integers summing to 100 which denote the number of troops you allocate to battlefield 1 through 10 respectively.
Your submission should also include a writeup explaining the reasoning for the strategy you chose.
The writeup should be a PDF with no more than 2 A4 pages.
We will review the written responses of candidates whose scores fall within the top quartile.
