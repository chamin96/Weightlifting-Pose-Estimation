OPTIMAL_KNEE_ANGLE = 165
OPTIMAL_BAR_ANGLE = 0

ERROR_MARGIN_KNEE_ANGLE = 10
ERROR_MARGIN_BAR_ANGLE = 30

WEIGHT_KNEE_SCORE = 7
WEIGHT_BAR_SCORE = 3

knee_score = 0
bar_score = 0
overall_score = 0

def kneeAngleScore(angle):
    error_factor = (abs(angle - OPTIMAL_KNEE_ANGLE)) / ERROR_MARGIN_KNEE_ANGLE
    knee_score = 10 - error_factor
    return knee_score

def barAngleScore(angle):
    error_factor = (abs(angle - OPTIMAL_BAR_ANGLE)) / ERROR_MARGIN_BAR_ANGLE
    bar_score = 10 - error_factor
    return bar_score

def overallScore(knee_score, bar_score):
    overall_score = (knee_score * WEIGHT_KNEE_SCORE + bar_score * WEIGHT_BAR_SCORE) / 10
    return overall_score