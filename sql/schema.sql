-- Example relational schema for storing machine readings.
-- Adjust data types and constraints for your chosen database engine.
CREATE TABLE machine_sensor_readings (
    machine_id VARCHAR(20) NOT NULL,
    machine_type CHAR(1) NOT NULL,
    air_temperature_k DECIMAL(7,2),
    process_temperature_k DECIMAL(7,2),
    rotational_speed_rpm INTEGER,
    torque_nm DECIMAL(7,2),
    tool_wear_min INTEGER,
    failure SMALLINT NOT NULL CHECK (failure IN (0, 1))
);
