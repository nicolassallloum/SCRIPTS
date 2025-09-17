-- PROCEDURE: suitedba.insert_update_object_bundle(jsonb, jsonb)
-- DROP PROCEDURE IF EXISTS suitedba.insert_update_object_bundle(jsonb, jsonb);


-- Author  : Nicolas.S
-- Created : 09/17/2025 10:00:00 AM
-- Version : V1

CREATE OR REPLACE PROCEDURE suitedba.insert_update_object_bundle(
	IN p_items jsonb,
	INOUT p_result jsonb DEFAULT NULL::jsonb)
LANGUAGE 'plpgsql'
AS $BODY$
DECLARE
    itm               JSONB;
    v_owner_name      TEXT;
    v_table_name      TEXT;
    v_order_no        INTEGER;
    v_user_id         INTEGER;
    v_object_id       BIGINT;
    v_table_id        BIGINT;
    v_column_group_id BIGINT;
    col_name          TEXT;
    col_id            BIGINT;
    deleted_count     INTEGER := 0;
    link_inserted_count INTEGER := 0;
    v_column_ids      BIGINT[] := '{}';
    v_column_type     INTEGER;
    v_has_new_columns BOOLEAN := false;
BEGIN
    -- Get first item from JSON array
    itm := p_items -> 0;

    -- Extract fields
    v_owner_name := itm ->> 'tableOwner';
    v_table_name := itm ->> 'tableName';
    v_order_no   := COALESCE((itm ->> 'orderNo')::INTEGER, 0);
    v_user_id    := COALESCE(
        (itm ->> 'createdBy')::INTEGER,
        (itm ->> 'updatedBy')::INTEGER
    );
    v_object_id  := (itm ->> 'objectId')::BIGINT;

    RAISE NOTICE '🔧 Owner: %, Table: %, Object ID: %, User: %, Order: %',
        v_owner_name, v_table_name, v_object_id, v_user_id, v_order_no;

    -- Validate
    IF v_object_id IS NULL THEN
        RAISE EXCEPTION 'objectId is required';
    END IF;

    -- ✅ VALIDATE OBJECT EXISTS IN cfg_object_def
    IF NOT EXISTS (
        SELECT 1
        FROM suitedba.cfg_object_def
        WHERE object_id = v_object_id
    ) THEN
        RAISE EXCEPTION 'Object ID % does not exist in cfg_object_def. Cannot link columns to non-existent object.', v_object_id;
    END IF;

    RAISE NOTICE '✅ Object % exists in cfg_object_def', v_object_id;

    -- Get or create table_id
    v_table_id := NULL;
    IF v_table_name IS NOT NULL AND v_owner_name IS NOT NULL THEN
        RAISE NOTICE '📥 Processing table: %.%', v_owner_name, v_table_name;

        INSERT INTO suitedba.cfg_table_config (created_by, table_name, table_owner)
        SELECT v_user_id, v_table_name, v_owner_name
        WHERE NOT EXISTS (
            SELECT 1 FROM suitedba.cfg_table_config
            WHERE table_owner = v_owner_name AND table_name = v_table_name
        );

        SELECT table_id
        INTO v_table_id
        FROM suitedba.cfg_table_config
        WHERE table_owner = v_owner_name AND table_name = v_table_name;

        IF v_table_id IS NULL THEN
            RAISE EXCEPTION 'Could not resolve table_id for %.%', v_owner_name, v_table_name;
        END IF;

        RAISE NOTICE '🔍 Table ID: %', v_table_id;
		
		    IF v_table_id IS NOT NULL THEN
      INSERT INTO suitedba.cfg_table_object_rel(
        table_id,
        object_id,
        order_no,
        created_by
      ) select 
        v_table_id,
        v_object_id,
        v_order_no,
        v_user_id
        
	   WHERE NOT EXISTS (
            SELECT 1 FROM suitedba.cfg_table_object_rel
            WHERE table_id = v_table_id AND object_id = v_object_id
        );

      RAISE NOTICE '🔗 Linked table_id % to object_id %', v_table_id, v_object_id;
    END IF;
		
		
    END IF;

    -- Handle operation type
    IF (itm ->> 'type') = 'saveNew' OR (itm ->> 'type') = 'update' THEN

        -- ============================
        -- DELETE specified column_ids (for "update" type)
        -- ============================
        IF (itm ->> 'type') = 'update' AND jsonb_typeof(itm -> 'deleted') = 'array' THEN
            WITH deleted AS (
                DELETE FROM suitedba.cfg_object_table_column_rel
                WHERE object_id = v_object_id
                  AND column_id IN (
                      SELECT value::BIGINT
                      FROM jsonb_array_elements_text(itm -> 'deleted') AS value
                  )
                RETURNING column_id
            )
            SELECT COUNT(*) INTO deleted_count FROM deleted;

            RAISE NOTICE '🗑️ Deleted % column links for object %', deleted_count, v_object_id;
        END IF;

        -- ============================
        -- ✅ INSERT COLUMNS FROM JSON + AUTO PK/NOT NULL (Combined Mode)
        -- ============================
        RAISE NOTICE '📋 Processing columns from input JSON + PK/NOT NULL columns';

        -- Collect column names from both sources
        FOR col_name, v_column_type IN
            -- Columns from JSON input + PK/NOT NULL columns
            SELECT column_name, column_type_code
            FROM (
                SELECT DISTINCT
                    c.column_name, 
                    CASE
                        WHEN c.data_type IN ('integer','smallint','bigint') OR c.udt_name IN ('int2','int4','int8') THEN 1
                        WHEN c.data_type IN ('numeric','real','double precision','decimal') THEN 2
                        WHEN c.data_type IN ('character varying','character','text') THEN 1
                        WHEN c.data_type IN ('date') THEN 6
                        WHEN c.data_type LIKE 'timestamp%' THEN 6
                        WHEN c.data_type = 'bytea' THEN 11
                        ELSE 1
                    END AS column_type_code,
                    c.ordinal_position  -- 👈 Included for ordering
                FROM (
                    -- Manual columns from JSON
                    SELECT jsonb_array_elements(
                        CASE
                            WHEN (itm ->> 'type') = 'saveNew' THEN itm -> 'columns'
                            WHEN (itm ->> 'type') = 'update' THEN
                                (SELECT jsonb_agg(jsonb_build_object('columnName', v))
                                 FROM jsonb_array_elements_text(itm -> 'inserted') AS v)
                            ELSE '[]'::jsonb
                        END
                    ) ->> 'columnName' AS column_name
                    UNION
                    -- Auto: PK or NOT NULL columns
                    SELECT c.column_name
                    FROM information_schema.columns c
                    WHERE c.table_schema = v_owner_name
                      AND c.table_name = v_table_name
                      AND (
                          EXISTS (
                              SELECT 1
                              FROM information_schema.table_constraints tc
                              JOIN information_schema.key_column_usage kcu 
                                  ON kcu.constraint_name = tc.constraint_name
                                  AND kcu.table_schema = tc.table_schema
                                  AND kcu.table_name = tc.table_name
                              WHERE tc.table_schema = c.table_schema
                                AND tc.table_name = c.table_name
                                AND tc.constraint_type = 'PRIMARY KEY'
                                AND kcu.column_name = c.column_name
                          )
                          OR c.is_nullable = 'NO'
                      )
                ) AS combined_columns
                JOIN information_schema.columns c 
                    ON c.table_schema = v_owner_name
                   AND c.table_name = v_table_name
                   AND c.column_name = combined_columns.column_name
                ORDER BY c.ordinal_position  -- ✅ Now allowed
            ) AS ordered_columns
        LOOP
            IF col_name IS NULL OR col_name = '' THEN
                CONTINUE;
            END IF;

            RAISE NOTICE '🔎 Processing column: %', col_name;

            -- Insert column if not exists
            INSERT INTO suitedba.cfg_column_config(
                column_id,
                column_desc,
                column_name,
                created_by,
                table_id,
                is_hyperlink,
                object_id,
                order_no,
                column_length,
                has_multiple_value,
                is_execution_suspended,
                cell_rendering,
                column_group_id,
                is_saved,
                input_lan_id,
                is_mandatory,
                is_unique_key,
                is_read_only,
                column_type_code
            )
            SELECT
                nextval('suitedba.s_column_config'),
                col_name,
                col_name,
                v_user_id,
                v_table_id,
                '0',
                NULL,
                v_order_no,
                100,
                '0',
                '0',
                '1',
                NULL,
                '0',
                13,
                0,
                '0',
                '0',
                v_column_type
            WHERE NOT EXISTS (
                SELECT 1
                FROM suitedba.cfg_column_config
                WHERE table_id = v_table_id
                  AND column_name = col_name
            );

            -- Get column_id
            SELECT column_id
            INTO col_id
            FROM suitedba.cfg_column_config
            WHERE table_id = v_table_id
              AND column_name = col_name;

            IF col_id IS NOT NULL THEN
                v_column_ids := array_append(v_column_ids, col_id);
                RAISE NOTICE '✅ Column "%" → ID %', col_name, col_id;
            ELSE
                RAISE NOTICE '❌ Failed to retrieve column_id for "%"', col_name;
            END IF;
        END LOOP;

        -- ✅ Only proceed if we have columns to link
        IF array_length(v_column_ids, 1) > 0 THEN

            -- Check if any column is not already linked
            SELECT EXISTS (
                SELECT 1
                FROM unnest(v_column_ids) AS cid
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM suitedba.cfg_object_table_column_rel otcr
                    WHERE otcr.object_id = v_object_id
                      AND otcr.column_id = cid
                )
            ) INTO v_has_new_columns;

            IF v_has_new_columns THEN
                -- ✅ Create new column group
                INSERT INTO suitedba.cfg_column_group(
                    column_group_desc,
                    order_no,
                    is_hidden,
                    has_access_add,
                    has_access_modify,
                    has_access_delete,
                    is_top_display,
                    is_read_only,
                    creation_date,
                    created_by,
                    column_group_code,
                    is_grid,
                    is_multi_selection,
                    is_advanced_search_applied,
                    is_multiple
                ) VALUES (
                    'Combined Group for Object ' || v_object_id,
                    0, '0', '0', '0', '0', '0', '0', NOW(), v_user_id, 0, '0', '0', '0', '0'
                )
                RETURNING column_group_id INTO v_column_group_id;

                RAISE NOTICE '🆕 Created new column_group_id: %', v_column_group_id;

                -- ✅ Link group to object
                INSERT INTO suitedba.cfg_column_group_object(
                    object_id,
                    column_group_id,
                    creation_date,
                    created_by
                ) VALUES (
                    v_object_id,
                    v_column_group_id,
                    NOW(),
                    v_user_id
                );

                RAISE NOTICE '🔗 Linked new column_group_id % to object_id %', v_column_group_id, v_object_id;

                -- Optional: Update column config group (if needed)
                UPDATE suitedba.cfg_column_config
                SET column_group_id = v_column_group_id
                WHERE table_id = v_table_id
                  AND column_id = ANY(v_column_ids)
                  AND (column_group_id IS NULL OR column_group_id != v_column_group_id);

            ELSE
                -- ❌ All columns already linked → reuse existing group
                SELECT column_group_id
                INTO v_column_group_id
                FROM suitedba.cfg_object_table_column_rel
                WHERE object_id = v_object_id
                  AND column_id = ANY(v_column_ids)
                LIMIT 1;

                IF v_column_group_id IS NULL THEN
                    -- Fallback: get any group for this object
                    SELECT column_group_id
                    INTO v_column_group_id
                    FROM suitedba.cfg_column_group_object
                    WHERE object_id = v_object_id
                    LIMIT 1;

                    IF v_column_group_id IS NULL THEN
                        RAISE EXCEPTION 'No column group found for object_id %, and no existing links to derive one.', v_object_id;
                    END IF;
                END IF;

                RAISE NOTICE 'ℹ️ All columns already linked. Reusing column_group_id: %', v_column_group_id;
            END IF;

            -- ✅ Insert into cfg_object_table_column_rel with auto-order
            DECLARE
                v_start_order INTEGER;
            BEGIN
                SELECT COALESCE(MAX(order_no), 0) + 1
                INTO v_start_order
                FROM suitedba.cfg_object_table_column_rel
                WHERE object_id = v_object_id;

                RAISE NOTICE '🔢 Starting order_no for new columns: %', v_start_order;

                WITH new_columns AS (
                    SELECT 
                        cid,
                        v_start_order + ROW_NUMBER() OVER () - 1 AS computed_order
                    FROM unnest(v_column_ids) AS cid
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM suitedba.cfg_object_table_column_rel otcr
                        WHERE otcr.object_id = v_object_id
                          AND otcr.column_id = cid
                    )
                ),
                inserted AS (
                    INSERT INTO suitedba.cfg_object_table_column_rel(
                        table_id,
                        object_id,
                        order_no,
                        column_group_id,
                        column_id,
                        created_by
                    )
                    SELECT
                        v_table_id,
                        v_object_id,
                        computed_order,
                        v_column_group_id,
                        cid,
                        v_user_id
                    FROM new_columns
                    RETURNING column_id
                )
                SELECT COUNT(*) INTO link_inserted_count FROM inserted;

                RAISE NOTICE '🔗 Inserted % new column links for object %', link_inserted_count, v_object_id;
            END;

        ELSE
            RAISE NOTICE 'ℹ️ No columns to process.';
        END IF;

    ELSE
        RAISE EXCEPTION 'Unsupported type: %', itm ->> 'type';
    END IF;

    -- Return result
    p_result := jsonb_build_object(
        'status', 'success',
        'message', 'Operation completed',
        'objectId', v_object_id,
        'columnGroupId', v_column_group_id,
        'columnsInserted', array_length(v_column_ids, 1),
        'linksInserted', link_inserted_count,
        'deletedCount', deleted_count,
        'affectedColumnIds', v_column_ids
    );

    RAISE NOTICE '🎉 Final result: %', p_result;

END;
$BODY$;
ALTER PROCEDURE suitedba.insert_update_object_bundle(jsonb, jsonb)
    OWNER TO postgres;

GRANT EXECUTE ON PROCEDURE suitedba.insert_update_object_bundle(jsonb, jsonb) TO PUBLIC;

GRANT EXECUTE ON PROCEDURE suitedba.insert_update_object_bundle(jsonb, jsonb) TO pgdba;

GRANT EXECUTE ON PROCEDURE suitedba.insert_update_object_bundle(jsonb, jsonb) TO pguser;

GRANT EXECUTE ON PROCEDURE suitedba.insert_update_object_bundle(jsonb, jsonb) TO postgres;

