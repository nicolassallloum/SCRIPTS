-- PROCEDURE: suitedba.insert_update_rule_object(jsonb, jsonb)
-- DROP PROCEDURE IF EXISTS suitedba.insert_update_rule_object(jsonb, jsonb);


-- Author  : Nicolas.S
-- Created : 09/17/2025 10:00:00 AM
-- Version : V1

CREATE OR REPLACE PROCEDURE suitedba.INSERT_UPDATE_RULE_OBJECT(
	IN p_items jsonb,
	INOUT p_result jsonb DEFAULT NULL::jsonb)
LANGUAGE 'plpgsql'
AS $BODY$
DECLARE
    item jsonb;
    v_object_id text;
    v_column_id text;
    v_json_data jsonb;
    v_updated_count int := 0;
BEGIN
    -- Loop through each item in the JSON array
    FOR item IN SELECT jsonb_array_elements(p_items)
    LOOP
        -- Extract fields
        v_object_id := item->>'objectId';
        v_column_id := item->>'columnId';
        v_json_data := item->'jsonData';

        -- Update only if row exists
        UPDATE suitedba.cfg_object_table_column_rel
        SET json_file_data = v_json_data
        WHERE object_id = v_object_id::numeric
          AND column_id = v_column_id::numeric;

        -- Check if any row was updated
        IF FOUND THEN
            v_updated_count := v_updated_count + 1;
        END IF;
    END LOOP;

    -- Return result
    p_result := jsonb_build_object(
        'status', 'success',
        'message', 'Update completed',
        'rows_updated', v_updated_count
    );

EXCEPTION
    WHEN OTHERS THEN
        p_result := jsonb_build_object(
            'status', 'error',
            'message', SQLERRM
        );
        RAISE;
END;
$BODY$;
ALTER PROCEDURE suitedba.insert_update_rule_object(jsonb, jsonb)
    OWNER TO postgres;

GRANT EXECUTE ON PROCEDURE suitedba.insert_update_rule_object(jsonb, jsonb) TO PUBLIC;

GRANT EXECUTE ON PROCEDURE suitedba.insert_update_rule_object(jsonb, jsonb) TO pgdba;

GRANT EXECUTE ON PROCEDURE suitedba.insert_update_rule_object(jsonb, jsonb) TO pguser;

GRANT EXECUTE ON PROCEDURE suitedba.insert_update_rule_object(jsonb, jsonb) TO postgres;

