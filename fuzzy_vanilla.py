# This is a partial implementation
# Contains functions for membership and defuzzification
# The rest is up to you


def triangular(x, a, b, c):
    return max(0, min((x - a) / (b - a), (c - x) / (c - b)))


def trapezoidal(x, a, b, c, d):
    return max(0, min((x - a) / (b - a), 1, (d - x) / (d - c)))


def s_shape(x, a, b):
    if x <= a:
        return 0
    elif x <= (a + b) / 2:
        return 2 * ((x - a) / (b - a)) ** 2
    elif x <= b:
        return 1 - 2 * ((x - b) / (b - a)) ** 2
    else:
        return 1


def z_shape(x, a, b):
    if x <= a:
        return 1
    elif a < x <= (a + b) / 2:
        return 1 - 2 * ((x - a) / (b - a)) ** 2
    elif (a + b) / 2 < x <= b:
        return 2 * ((x - b) / (b - a)) ** 2
    else:
        return 0


# Fuzzification
def fuzzification_conditions(condition):
    return {
        'slight': z_shape(condition, 1, 1.25),
        'intermediate': trapezoidal(condition, 1, 1.25, 2.5, 3),
        'rough': s_shape(condition, 2.5, 3)
    }


# Other fuzzification functions go here

def inference(conditions_fuzzy, visibility_fuzzy):
    # If visibility is bad or conditions are rough – speed is low.
    rule1 = max(conditions_fuzzy["rough"], visibility_fuzzy["bad"])

    # Apply other rules here


# #### For defuzzification!
def aggregated_membership(x, fuzzy_strengths):
    return max(
        fuzzy_strengths[0] * z_shape(x, 15, 20),
        fuzzy_strengths[1] * trapezoidal(x, 15, 30, 40, 50),
        fuzzy_strengths[2] * s_shape(x, 45, 50)
    )


def defuzzification(mf, fuzzy_strengths, x_min, x_max, step_size=0.5):
    x_values = [x_min + i * step_size for i in range(int((x_max - x_min) / step_size) + 1)]
    mu_values = [mf(x, fuzzy_strengths) for x in x_values]
    numerator = sum(x * mu for x, mu in zip(x_values, mu_values))
    denominator = sum(mu_values)
    return numerator / denominator if denominator != 0 else 0


# #### End of defuzzification

if __name__ == "__main__":
    # Create your crisp values

    # Fuzzify your values

    # Apply rules to create new fuzzy numbers

    # Call the defuzzification function to calculate speed

    pass  # DELETE THIS LINE