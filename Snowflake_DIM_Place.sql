CREATE TABLE IF NOT EXISTS DIM_PLACE (
ID_PLACE INT,
PLACE_NAME VARCHAR,
PLACE_CATEGORY VARCHAR,
INSERTED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

INSERT INTO DIM_PLACE (
ID_PLACE, PLACE_NAME, PLACE_CATEGORY
)
VALUES (-1, 'unknown', 'unknown');

INSERT INTO DIM_PLACE (
SELECT 
    "Place_ID" AS ID_PLACE,
    "Place_Name" AS PLACE_NAME,
    "Place_Category" AS PLACE_CATEGORY,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
FROM
    "place"
    );

UPDATE FACT_SPELL
SET ID_PLACE = CASE
		WHEN ID_PLACE = 74 THEN REPLACE(ID_PLACE, 74, 68)
                 ELSE ID_PLACE
            		END;