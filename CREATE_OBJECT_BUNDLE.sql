-- PROCEDURE: suitedba.create_object_bundle(jsonb, jsonb)
-- DROP PROCEDURE IF EXISTS suitedba.create_object_bundle(jsonb, jsonb);


-- Author  : Nicolas.S
-- Created : 09/17/2025 10:00:00 AM
-- Version : V1

CREATE OR REPLACE PROCEDURE suitedba.create_object_bundle(
	IN p_items jsonb,
	INOUT p_result jsonb DEFAULT NULL::jsonb)
LANGUAGE 'plpgsql'
AS $BODY$
DECLARE
  itm               jsonb;
  v_client_id       integer;
  v_owner_name      text;
  v_table_name      text;
  v_menu_name       text;
  v_user_id         integer;

  v_object_id       bigint;
  v_column_group_id bigint;
  v_table_id        bigint;

  v_order           integer;
  v_col_order       integer;

  -- helpers
  v_has_table       boolean;

  -- For final JSON output
  v_fieldset_id     bigint;
  v_id              bigint;
  v_column_ids      bigint[] := '{}';
  v_new_column_ids_temp  bigint[]; 
  v_debug_text      text;  -- ✅ FOR DEBUG OUTPUT

BEGIN
  IF p_items IS NULL OR jsonb_typeof(p_items) <> 'array' THEN
    RAISE EXCEPTION 'p_items must be a JSON array';
  END IF;

  -- Loop over array
  FOR itm IN SELECT * FROM jsonb_array_elements(p_items)
  LOOP
    -- ✅ RESET VARIABLES FOR EACH ITEM
    v_owner_name := NULL;
    v_table_name := NULL;
    v_table_id := NULL;
    v_has_table := false;
    v_object_id := NULL;
    v_column_group_id := NULL;
    v_new_column_ids_temp := '{}';
    v_debug_text := NULL;

    -- Extract inputs
    v_client_id  := (itm->>'clientId')::int;
    v_owner_name := nullif(itm->>'ownerName','');
    v_table_name := nullif(itm->>'tableName','');
    v_menu_name  := nullif(itm->>'menuName','');
    v_user_id    := (itm->>'userId')::int;

    IF v_menu_name IS NULL THEN
      RAISE EXCEPTION 'menuName is required in each item';
    END IF;

    v_has_table := (v_owner_name IS NOT NULL AND v_table_name IS NOT NULL);

    -- ✅ SET SEARCH PATH SO information_schema CAN SEE THE TABLE
    IF v_owner_name IS NOT NULL THEN
        EXECUTE 'SET search_path TO ' || quote_ident(v_owner_name) || ', public, suitedba';
        RAISE NOTICE '🔧 Set search_path to: %, public, suitedba', v_owner_name;
    END IF;

    ------------------------------------------------------------------------
    -- 1) cfg_object_def
    ------------------------------------------------------------------------
    INSERT INTO suitedba.cfg_object_def(
      client_id,
      object_name,
      object_type,
      is_main,
      created_by,
      has_full_display,
      tech_is_grid,
      is_tree_grid,
      is_form_flip,
      is_am_load,
      is_advanced_hidden,
      tech_is_query_form,
      has_view,
      has_multiple_selection,
      is_dynamic_report,
      has_advanced_search,
      advanced_search_procedure_name,
      api_function_name,
      is_api_enabled,
      is_dynamic_title_enabled,
      order_no,
      hide_grid,
      no_refresh_on_close,
      can_add_editable,
      is_mapping_form,
      show_editable_search,
      can_modify_editable,
      can_delete_editable,
      show_editable_top,
      isformulavisible,
      is_favorite,
      is_published,
      tech_has_drill_down,
      is_disabled,
      is_cql,
      is_default,
      is_row_grouping,
      is_row_group_hidden,
      form_view,
      from_api,
      from_procedure,
      is_tree_grid_editable,
      add_mode_qbe_id,
      modify_mode_qbe_id,
      delete_mode_qbe_id,
      import_mode_qbe_id,
      qbe_id,
      save_condition_qbe_id,
      dynamic_title_name
    ) VALUES (
      v_client_id,                 -- client_id
      v_menu_name,                 -- object_name
      16,                          -- object_type
      1,                           -- is_main
      v_user_id,                   -- created_by
      '0',                         -- has_full_display
      1,                           -- tech_is_grid
      0,                           -- is_tree_grid
      0,                           -- is_form_flip
      0,                           -- is_am_load
      0,                           -- is_advanced_hidden
      '0',                         -- tech_is_query_form
      '0',                         -- has_view
      '0',                         -- has_multiple_selection
      '0',                         -- is_dynamic_report
      0,                           -- has_advanced_search
      NULL,                        -- advanced_search_procedure_name
      'callRestApi2',              -- api_function_name
      '0',                         -- is_api_enabled
      '0',                         -- is_dynamic_title_enabled
      1,                           -- order_no
      '0',                         -- hide_grid
      '0',                         -- no_refresh_on_close
      '0',                         -- can_add_editable
      '0',                         -- is_mapping_form
      NULL,                        -- show_editable_search (dynamicTitleName)
      '0',                         -- can_modify_editable
      '0',                         -- can_delete_editable
      '0',                         -- show_editable_top
      '0',                         -- isformulavisible
      '0',                         -- is_favorite
      '0',                         -- is_published
      '0',                         -- tech_has_drill_down
      '0',                         -- is_disabled
      '0',                         -- is_cql
      '0',                         -- is_default
      '0',                         -- is_row_grouping
      '0',                         -- is_row_group_hidden
      '0',                         -- form_view
      '0',                         -- from_api
      '0',                         -- from_procedure
      '0',                         -- is_tree_grid_editable
      NULL,                        -- add_mode_qbe_id
      NULL,                        -- modify_mode_qbe_id
      NULL,                        -- delete_mode_qbe_id
      NULL,                        -- import_mode_qbe_id
      NULL,                        -- qbe_id
      NULL,                        -- save_condition_qbe_id
      v_menu_name
    )
    RETURNING object_id INTO v_object_id;

    RAISE NOTICE '✅ Created object_id: % for menu: %', v_object_id, v_menu_name;

    ------------------------------------------------------------------------
    -- 2) cfg_column_group
    ------------------------------------------------------------------------
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
      'fielSet',      -- column_group_desc
      0,              -- order_no
      '0',            -- is_hidden
      '0',            -- has_access_add
      '0',            -- has_access_modify
      '0',            -- has_access_delete
      '0',            -- is_top_display
      '0',            -- is_read_only
      now(),          -- creation_date
      v_user_id,      -- created_by
      0,              -- column_group_code
      '0',            -- is_grid
      '0',            -- is_multi_selection
      '0',            -- is_advanced_search_applied
      '0'             -- is_multiple
    )
    RETURNING column_group_id INTO v_column_group_id;

    RAISE NOTICE '✅ Created column_group_id: %', v_column_group_id;

    ------------------------------------------------------------------------
    -- 3) cfg_column_group_object (link group ↔ object)
    ------------------------------------------------------------------------
    INSERT INTO suitedba.cfg_column_group_object(
      object_id,
      column_group_id,
      creation_date,
      created_by
    ) VALUES (
      v_object_id,
      v_column_group_id,
      now(),
      v_user_id
    );

    RAISE NOTICE '🔗 Linked object_id % to column_group_id %', v_object_id, v_column_group_id;

    ------------------------------------------------------------------------
    -- 4) cfg_table_config (insert-once by owner+name)
    ------------------------------------------------------------------------
    v_table_id := NULL;
	
    IF v_has_table THEN
      RAISE NOTICE '📥 Processing table: %.%', v_owner_name, v_table_name;

      INSERT INTO suitedba.cfg_table_config (
        created_by,
        table_name,
        table_owner
      )
      SELECT 
        v_user_id,
        v_table_name,
        v_owner_name
      WHERE NOT EXISTS (
        SELECT 1
        FROM suitedba.cfg_table_config t
        WHERE t.table_owner = v_owner_name
          AND t.table_name  = v_table_name
      );

      SELECT t.table_id
      INTO v_table_id
      FROM suitedba.cfg_table_config t
      WHERE t.table_owner = v_owner_name
        AND t.table_name  = v_table_name;

      RAISE NOTICE '🔍 Found v_table_id: %', v_table_id;
    END IF;

    ------------------------------------------------------------------------
    -- 5) cfg_table_object_rel (link table ↔ object)
    ------------------------------------------------------------------------
    IF v_table_id IS NOT NULL THEN
      INSERT INTO suitedba.cfg_table_object_rel(
        table_id,
        object_id,
        order_no,
        created_by
      ) VALUES (
        v_table_id,
        v_object_id,
        0,
        v_user_id
      )
      ON CONFLICT DO NOTHING;

      RAISE NOTICE '🔗 Linked table_id % to object_id %', v_table_id, v_object_id;
    END IF;

    ------------------------------------------------------------------------
    -- 6 & 7) Insert and Link columns that are PRIMARY KEY OR NOT NULL
    ------------------------------------------------------------------------
    IF v_table_id IS NOT NULL THEN
      RAISE NOTICE '🎯 Processing columns for table_id: %', v_table_id;

      v_col_order := 0;

      -- ============================
      -- INSERT COLUMNS THAT ARE EITHER PRIMARY KEY OR NOT NULL
      -- ============================
      v_new_column_ids_temp := '{}'; -- Reset array

      WITH inserted AS (
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
              readonly_qbe_id,
              default_qbe_id,
              mandatory_qbe_id,
              qbe_id,
              dependent_qbe_id,
              is_default,
              is_output,
              is_grouping_enabled,
              is_pivot_enabled,
              is_value_enabled,
              tech_is_system_table,
              is_grid,
              is_form,
              is_unique_key,
              is_read_only,
              has_background_action,
              is_link,
              is_editable_tree_grid,
              is_grid_form_insert,
              is_lookup_tree_grid,
              is_list,
              is_hidden,
              column_type_code
          )
          SELECT
              nextval('SUITEDBA.S_COLUMN_CONFIG'),
              c.column_name AS column_desc,
              c.column_name AS column_name,
              v_user_id,
              v_table_id,
              '0' AS is_hyperlink,
              NULL AS object_id,
              row_number() OVER (ORDER BY c.ordinal_position) - 1 + v_col_order AS order_no,
              100 AS column_length,
              '0' AS has_multiple_value,
              '0' AS is_execution_suspended,
              '1' AS cell_rendering,
              v_column_group_id,
              '0' AS is_saved,
              13 AS input_lan_id,
              1 AS is_mandatory,
              NULL, NULL, NULL, NULL, NULL,
              '0' AS is_default, '0' AS is_output,
              '0' AS is_grouping_enabled, '0' AS is_pivot_enabled, '0' AS is_value_enabled,
              '0' AS tech_is_system_table,
              '0' AS is_grid, '0' AS is_form,
              CASE
                  WHEN EXISTS (
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
                  ) THEN '1'
                  ELSE '0'
              END AS is_unique_key,
              '0' AS is_read_only,
              '0' AS has_background_action,
              '0' AS is_link,
              '0' AS is_editable_tree_grid,
              '0' AS is_grid_form_insert,
              '0' AS is_lookup_tree_grid,
              '0' AS is_list,
              '0' AS is_hidden,
              CASE
                  WHEN c.data_type IN ('integer','smallint','bigint') OR c.udt_name IN ('int2','int4','int8') THEN 1
                  WHEN c.data_type IN ('numeric','real','double precision','decimal') THEN 2
                  WHEN c.data_type IN ('character varying','character','text') THEN 1
                  WHEN c.data_type IN ('date') THEN 6
                  WHEN c.data_type LIKE 'timestamp%' THEN 6
                  WHEN c.data_type = 'bytea' THEN 11
                  ELSE 1
              END AS column_type_code
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
            AND NOT EXISTS (
                SELECT 1
                FROM suitedba.cfg_column_config cc
                WHERE cc.table_id = v_table_id
                  AND cc.column_name = c.column_name
            )
          RETURNING column_id
      )
      SELECT array_agg(column_id) INTO v_new_column_ids_temp
      FROM inserted;

      v_column_ids := v_column_ids || COALESCE(v_new_column_ids_temp, '{}');

      -- Update v_col_order after insert
      SELECT COALESCE(MAX(order_no), -1) + 1
      INTO v_col_order
      FROM suitedba.cfg_column_config
      WHERE table_id = v_table_id;

      -- DEBUG: How many columns exist now in cfg_column_config for this table?
      RAISE NOTICE '📊 cfg_column_config now has % columns for table_id %', 
          (SELECT COUNT(*) FROM suitedba.cfg_column_config WHERE table_id = v_table_id),
          v_table_id;

      -- List column names inserted
      IF array_length(v_new_column_ids_temp, 1) > 0 THEN
          RAISE NOTICE '🆕 Inserted column_ids: %', v_new_column_ids_temp;
      ELSE
          RAISE NOTICE 'ℹ️ No new columns inserted (already exist?)';
      END IF;

      ------------------------------------------------------------------------
      -- ✅ DEBUG: Check what columns are qualifying (PK or NOT NULL) in information_schema
      ------------------------------------------------------------------------
      RAISE NOTICE '🔍 Checking qualifying columns in information_schema for %.%', v_owner_name, v_table_name;

      WITH table_info AS (
          SELECT table_owner, table_name
          FROM suitedba.cfg_table_config
          WHERE table_id = v_table_id
      ),
      qualifying_columns AS (
          SELECT c.column_name,
                 c.is_nullable,
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
                 ) AS is_pk
          FROM information_schema.columns c
          CROSS JOIN table_info ti
          WHERE c.table_schema = ti.table_owner
            AND c.table_name = ti.table_name
      )
      SELECT string_agg(
          column_name || '(' || is_nullable || ',' || is_pk || ')',
          ', '
      ) INTO v_debug_text
      FROM qualifying_columns;

      IF v_debug_text IS NOT NULL AND v_debug_text != '' THEN
          RAISE NOTICE '📋 All columns in table: %', v_debug_text;
      ELSE
          RAISE NOTICE '❌ NO COLUMNS FOUND in information_schema for %.% — PERMISSION/SCHEMA ISSUE?', v_owner_name, v_table_name;
      END IF;

      -- Now do the real insert into cfg_object_table_column_rel
      WITH table_info AS (
          SELECT table_owner, table_name
          FROM suitedba.cfg_table_config
          WHERE table_id = v_table_id
      ),
      qualifying_columns AS (
          SELECT c.column_name
          FROM information_schema.columns c
          CROSS JOIN table_info ti
          WHERE c.table_schema = ti.table_owner
            AND c.table_name = ti.table_name
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
      )
      INSERT INTO suitedba.cfg_object_table_column_rel(
        table_id,
        object_id,
        order_no,
        column_group_id,
        column_id,
        CREATED_BY
      )
      SELECT
        cc.table_id,
        v_object_id,
        cc.order_no,
        v_column_group_id,
        cc.column_id,
        v_user_id
      FROM suitedba.cfg_column_config cc
      WHERE cc.table_id = v_table_id
        AND cc.column_name IN (SELECT column_name FROM qualifying_columns)
        AND NOT EXISTS (
          SELECT 1
          FROM suitedba.cfg_object_table_column_rel otcr
          WHERE otcr.table_id = cc.table_id
            AND otcr.object_id = v_object_id
            AND otcr.column_id = cc.column_id
        );

      GET DIAGNOSTICS v_col_order = ROW_COUNT;
      RAISE NOTICE '✅ Inserted % rows into cfg_object_table_column_rel for object_id %', v_col_order, v_object_id;
	
      IF v_col_order = 0 THEN
          RAISE NOTICE '⚠️ WARNING: No rows inserted. Possible causes:';
          RAISE NOTICE '   - Columns not found in cfg_column_config (check column_name case)';
          RAISE NOTICE '   - Columns already linked to this object';
          RAISE NOTICE '   - qualifying_columns returned no matches (PK or NOT NULL)';
      END IF;
	
	--insert the columns into v_columns 
	 WITH table_info AS (
          SELECT table_owner, table_name
          FROM suitedba.cfg_table_config
          WHERE table_id = v_table_id
      ),
      qualifying_columns AS (
          SELECT c.column_name
          FROM information_schema.columns c
          CROSS JOIN table_info ti
          WHERE c.table_schema = ti.table_owner
            AND c.table_name = ti.table_name
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
      )
	SELECT array_agg(cc.column_id) into v_column_ids       
	FROM suitedba.cfg_column_config cc
      WHERE cc.table_id = v_table_id
        AND cc.column_name IN (SELECT column_name FROM qualifying_columns);
		
		RAISE NOTICE 'NEW COLUMNS IDS %',v_column_ids;
	
      -- Final check
      RAISE NOTICE '📈 Total links for object_id %: %', 
          v_object_id,
          (SELECT COUNT(*) FROM suitedba.cfg_object_table_column_rel WHERE object_id = v_object_id);

    END IF; -- v_table_id

    -- Capture for final output (overwrite each iteration — last one wins)
    v_fieldset_id := v_column_group_id;
    v_id          := v_object_id;

  END LOOP;

  -- Build and return final JSON response via INOUT parameter
  p_result := jsonb_build_object(
    'data', jsonb_build_object(
      'code', '0',
      'status', 'success',
      'description', 'SAVED SUCCESSFULLY',
      'userId', v_user_id,
      'objectId', v_object_id,
      'jsonRes', NULL,
      'showSaveButton', 0,
      'menuCode', NULL,
      'fieldsetId', v_fieldset_id,
      'columnIds', v_column_ids,
      'id', v_id
    )
  );

  RAISE NOTICE '🎉 Final result: %', p_result;

END;
$BODY$;
ALTER PROCEDURE suitedba.create_object_bundle(jsonb, jsonb)
    OWNER TO postgres;

GRANT EXECUTE ON PROCEDURE suitedba.create_object_bundle(jsonb, jsonb) TO PUBLIC;

GRANT EXECUTE ON PROCEDURE suitedba.create_object_bundle(jsonb, jsonb) TO pgdba;

GRANT EXECUTE ON PROCEDURE suitedba.create_object_bundle(jsonb, jsonb) TO pguser;

GRANT EXECUTE ON PROCEDURE suitedba.create_object_bundle(jsonb, jsonb) TO postgres;

