-- CREATE TABLE IF NOT EXISTS annotations (
--     image_id int GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
--     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
-- );

-- ALTER TABLE annotations 
-- ADD COLUMN class VARCHAR(255) NOT NULL;

-- DROP TABLE annotations;

-- SELECT * FROM annotations;

-- INSERT INTO annotations (class) VALUES ('person holding a book');

-- UPDATE annotations
-- SET class = 'person dropping a book'
-- WHERE image_id = 1;

-- DELETE FROM annotations
-- WHERE image_id = 1;

-- SELECT * FROM annotations;

-- CREATE TABLE users (
--     user_id int GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
--     image_id int REFERENCES annotations(image_id) ON DELETE CASCADE,
--     title text NOT NULL
-- )

-- INSERT INTO users (image_id, title) VALUES (1, 'Sample Annotation');

-- SELECT * FROM users INNER JOIN annotations ON users.image_id = annotations.image_id;

-- CREATE TABLE image_annotations (
--     id SERIAL PRIMARY KEY,
--     uploaded_image_url TEXT,
--     class TEXT,
--     person_class TEXT,
--     object_class TEXT,
--     person_bounding_box FLOAT[],
--     object_bounding_box FLOAT[],
--     interaction_bounding_box FLOAT[],
--     data_annotation_sentences TEXT[],
--     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
-- );