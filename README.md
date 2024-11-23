# UFC Ranking System by Fight Minds

## General Explanation

### How to Enter the Ranking:

- **The first victory in a weight class determines the position the athlete takes.**

### How to Earn Points:

- **By being active**: A draw earns a few points, and a victory earns a lot of points.
- **Winning against a higher-ranked opponent** allows you to gain a portion of their points, plus a bonus based on the method and other criteria (bonuses, streaks, etc.).
- **Winning against a lower-ranked opponent** allows you to earn points based on the method of victory and other criteria (bonuses, streaks, etc.).

### How to Lose Points:

- **A defeat causes the loss of some points.**
- **Points decrease for older fights.**

---

## Detailed Explanation

### Initialization of Points

- **Base Points**: Every fighter starts with a base score of `0.01`.

### Adjustment and Erosion Coefficients

- **Expiration Coefficient (Per month)**: Points adjust based on the time elapsed since the last fight.

![Expiration Coefficient Chart](/chart.png)

---

### Winning and Losing Points

#### Point Gain:

Points gained during a fight depend on several factors:

- **Victory Coefficients**:

  - KO/TKO or submission: `1.2`
  - Unanimous decision: `1.5`
  - Split or majority decision: `1.1`
  - Disqualification (DQ): `1.0`

- **Performance Bonus**:

  - "Performance of the Night": `1.3` times the base score.
  - "Fight of the Night": `1.3` times the base score.

- **Title on the Line**:

  - Title fights apply a multiplier coefficient of `1.5`.

- **Opponent Bonus**:

  - Winner earns 50% of their opponent's points. If this does not allow them to surpass the opponent in ranking, their points are adjusted to their opponent's points + `0.01`.

- **Winning Streak**:

  - Each consecutive victory increases a multiplier coefficient of `1 + (0.005 * n)`, where `n` is the number of consecutive victories.

- **Gain**:

  - To encourage activity, the winner receives `50%` of the average points of the category for each victory.

- **Draw**:

  - Fighters earn `20%` of the class average points and `5%` of their own total points.

- **No Contest**:

  - Fighters earn `5%` of the class average points.

- **5-Round Fight**:

  - A 5-round fight provides a bonus coefficient of `1.10`.

- **Name Value Bonus**:

  - Former champions earn a bonus coefficient of `1.5` for subsequent fights in the same weight class.

- **Missed Weight**:
  - Winners who missed weight earn `20%` fewer points.

---

#### Point Loss:

When a fighter loses, they incur a loss of points influenced by:

- **Loss Streak**:

  - Each consecutive loss increases a multiplier coefficient of `1 + (0.005 * n)`, where `n` is the number of consecutive losses.

- **Loss**:

  - Fighters lose `10%` of the average points of the category.

- **Title Fight**:

  - Losing a title fight divides the point loss by `1.3`.

- **Missed Weight**:
  - Losers who missed weight lose an additional `20%` of their points.

---

### Example Fight: September 28, 2024

#### Analyzed Fight: Morgan Charriere vs. Gabriel Miranda

- **Winner**: Morgan Charriere
- **Method of Victory**: KO/TKO (`1.2` bonus points)
- **Category**: Featherweight
- **Performance of the Night**: Yes (`1.3` bonus points)

---

#### Score Calculation Before the Fight:

1. **Morgan Charriere**:
   - Morgan's previous score in Featherweight = `0.037`.
   - Expiration of the score (based on seniority) = `1.0`.
   - **Adjusted score before the fight**: `0.037 × 1 = 0.037`.

---

#### Winner's Score After the Fight:

1. **Morgan's Victory Calculation**:

   - KO Bonus = `1.2`.
   - Performance Bonus = `1.3`.
   - Victory Boost = `1 + (0 × 0.005) = 1.0`.
   - Average points of category: `0.021`.
   - Morgan earns `50%` of opponent's points × bonuses: `(0.0219 × 1.2 + 0.021) × 1.3 = 0.0614`.

2. **New Score**:
   - Morgan's new score = `0.037 + 0.0614 = 0.0984`.

---

#### Effect on the Opponent:

1. **Gabriel Miranda**:
   - Gabriel's previous score: `0.0439`.
   - Loss Penalty = `20%` of category average points × method of loss.
   - New Score = `0.0439 - 0.0105 = 0.0334`.

---

## Key Takeaways:

1. Rewards active fighters while penalizing inactivity.
2. Values significant victories (finishes, title fights) and performance bonuses.
3. Accounts for streaks (winning or losing).
