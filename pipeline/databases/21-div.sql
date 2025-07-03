-- creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student.
DELIMITER //

DROP FUNCTION IF EXISTS SafeDiv;

CREATE FUNCTION SafeDiv (
    a INT,
    b INT
)
RETURNS FLOAT
DETERMINISTIC
BEGIN
    DECLARE c FLOAT;

    IF b = 0 THEN
        SET c = 0;
    ELSE
        SET c = a / b;
    END IF;

    RETURN c;
END; //

DELIMITER ;
