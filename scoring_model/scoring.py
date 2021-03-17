OPTIMAL_KNEE_ANGLE = 82
OPTIMAL_BAR_ANGLE = 0
OPTIMAL_LIMB_ANGLE = 180

ERROR_MARGIN_KNEE_ANGLE = 10
ERROR_MARGIN_BAR_ANGLE = 2
ERROR_MARGIN_LIMB_ANGLE = 5

WEIGHT_KNEE_SCORE = 5
WEIGHT_BAR_SCORE = 3
WEIGHT_ARMS_SCORE = 1
WEIGHT_LEGS_SCORE = 1

knee_score = 0
bar_score = 0
arms_score = 0
legs_score = 0
overall_score = 0


def kneeAngleScore(angle):
    error_factor = (abs(angle - OPTIMAL_KNEE_ANGLE)) / ERROR_MARGIN_KNEE_ANGLE
    knee_score = 10 - error_factor
    return knee_score


def barAngleScore(angle):
    error_factor = (abs(angle - OPTIMAL_BAR_ANGLE)) / ERROR_MARGIN_BAR_ANGLE
    bar_score = 10 - error_factor
    return bar_score


def armsAngleScore(angle1, angle2):
    error_factor = (
        abs(angle1 - OPTIMAL_LIMB_ANGLE) + abs(angle2 - OPTIMAL_LIMB_ANGLE)
    ) / (2 * ERROR_MARGIN_LIMB_ANGLE)
    arms_score = 10 - error_factor
    return arms_score


def legsAngleScore(angle1, angle2):
    error_factor = (
        abs(angle1 - OPTIMAL_LIMB_ANGLE) + abs(angle2 - OPTIMAL_LIMB_ANGLE)
    ) / (2 * ERROR_MARGIN_LIMB_ANGLE)
    legs_score = 10 - error_factor
    return legs_score


def overallScore(knee_score, bar_score, legs_score, arms_score):
    overall_score = (
        knee_score * WEIGHT_KNEE_SCORE
        + bar_score * WEIGHT_BAR_SCORE
        + legs_score * WEIGHT_LEGS_SCORE
        + arms_score * WEIGHT_ARMS_SCORE
    ) / 10
    return overall_score