# The story of the idea

Salvatore Pennacchio, October 2026.

The problem is easy to state and hard to solve: compute the variance of
float32 numbers that all sit around a very large value (a billion, a
million million) and that deviate from it by a very small amount. The
textbook formula, the one that subtracts two almost equal numbers, loses
the digits of the difference and returns a wrong number. The computer,
with its 24 bits of mantissa, cannot hold it.

The solution came out of a conversation with my son, and I tell it here
the way I would tell it to him.

## Step 1: make the numbers small

The first idea was: "If the numbers are too large, divide them by a
thousand, do the arithmetic on the small numbers, and put things back at
the end."

This idea lights the lamp: to keep the precision, the giant numbers have
to be made small before the variance is computed. But dividing by a
thousand only shifts the decimal point: it does not free space in the
bits. The large part of the number still fills the mantissa, and the
digits of the difference are still outside. A stronger way to remove the
large part is needed.

## Step 2: remove the giant with the shift

The second idea was: "Use a letter for the large part, the way the
computer writes 1e14."

From there the step is this: instead of an abstract letter, use the first
number of the row as the "letter", and subtract it from every number.
This is the shift:

    y = x - x[0]

Subtracting that value, the large part disappears for every number, and
only the small difference is left. If the numbers were all identical,
they collapse to exactly `0.0`. The large part is gone.

## Step 3: split the two pieces so the remainders are not lost

The third idea was: "Large computations can be split into small and
independent pieces, so the remainders are not lost."

This is the part that becomes the final formula in the code. The variance
can be written in two ways that look the same and are not:

    var = (1/n) sum (x_i - mean)^2
    var = (1/n) sum x_i^2 - mean^2

The second form is unstable, because it subtracts two almost equal
numbers (`sum x_i^2` and `n * mean^2`). But if, instead of applying it to
the original numbers, we apply it to the numbers already shifted
`y = x - x[0]` and centred a second time `d = y - mean(y)`, the two terms

    sum(d^2)         and         (sum(d))^2 / n

share the same mean `m`. Their rounding error on `m` is the same in the
two terms, and **it cancels in the subtraction**. There is no need to
compute `m` in double precision: it is enough to see that the error is the
same, and to subtract it.

## The final formula

Putting together the shift of Step 2 and the cancellation of Step 3:

    M2 = max(0, sum(d^2) - (sum(d))^2 / n)

with `d = y - mean(y)` and `y = x - x[0]`. The variance is `M2 / (n - ddof)`.

This is the formula in `dense_armor/utility/stats/stable.py`, and the
numbers that verify it are in `benchmarks/stable_variance_float32.json`.
On the four datasets of the task the maximum relative error is between
`1.1e-7` and `1.7e-7`, below the `2e-7` threshold, and on identical values
the result is exactly `0.0`.
