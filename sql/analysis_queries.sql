-- 1. Overall failure rate
SELECT
    COUNT(*) AS total_readings,
    SUM(failure) AS failures,
    1.0 * SUM(failure) / COUNT(*) AS failure_rate
FROM machine_sensor_readings;

-- 2. Failure rate by machine type
SELECT
    machine_type,
    COUNT(*) AS total_readings,
    SUM(failure) AS failures,
    1.0 * SUM(failure) / COUNT(*) AS failure_rate
FROM machine_sensor_readings
GROUP BY machine_type
ORDER BY failure_rate DESC;

-- 3. Average sensor values by failure outcome
SELECT
    failure,
    AVG(air_temperature_k) AS avg_air_temperature_k,
    AVG(process_temperature_k) AS avg_process_temperature_k,
    AVG(rotational_speed_rpm) AS avg_speed_rpm,
    AVG(torque_nm) AS avg_torque_nm,
    AVG(tool_wear_min) AS avg_tool_wear_min
FROM machine_sensor_readings
GROUP BY failure;

-- 4. Readings with high tool wear (illustrative threshold)
SELECT *
FROM machine_sensor_readings
WHERE tool_wear_min >= 200
ORDER BY tool_wear_min DESC;
