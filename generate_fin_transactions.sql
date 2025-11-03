CREATE OR REPLACE PROCEDURE SSDX_ENG.generate_fin_transactions(
    p_rows      integer,
    p_from_date date   DEFAULT date_trunc('year', current_date)::date,
    p_to_date   date   DEFAULT current_date
)
LANGUAGE plpgsql
AS $$
DECLARE
  v_src_ids bigint[];  -- VASP ids
  v_dst_ids bigint[];  -- CLIENT_VASP ids
  v_src_cnt int;
  v_dst_cnt int;
BEGIN
  IF p_rows IS NULL OR p_rows <= 0 THEN
    RAISE EXCEPTION 'p_rows must be > 0';
  END IF;

  IF p_from_date > p_to_date THEN
    RAISE EXCEPTION 'p_from_date (%) must be <= p_to_date (%)', p_from_date, p_to_date;
  END IF;

  /* Load pools once into arrays (fast + truly random per row) */
  SELECT array_agg(rc.customer_id) INTO v_src_ids
  FROM sdedba.ref_customer rc
  WHERE rc.customer_aka = 'VASP';

  SELECT array_agg(rc.customer_id) INTO v_dst_ids
  FROM sdedba.ref_customer rc
  WHERE rc.customer_aka = 'VASP CLIENT';

  v_src_cnt := COALESCE(array_length(v_src_ids, 1), 0);
  v_dst_cnt := COALESCE(array_length(v_dst_ids, 1), 0);

  IF v_src_cnt = 0 THEN
    RAISE EXCEPTION 'No SRC customers found with customer_aka = %', 'VASP';
  END IF;
  IF v_dst_cnt = 0 THEN
    RAISE EXCEPTION 'No DST customers found with customer_aka = %', 'VASP CLIENT';
  END IF;

  /* Insert p_rows transactions */
  WITH
  gs AS (
    SELECT generate_series(1, p_rows) AS n
  ),
  picked AS (
    SELECT
      n,
      (p_from_date + ((random() * ((p_to_date - p_from_date) + 1))::int))::date AS tran_date,
      /* random indexes into arrays: 1..v_src_cnt, 1..v_dst_cnt */
      (floor(random() * v_src_cnt)::int + 1) AS src_idx,
      (floor(random() * v_dst_cnt)::int + 1) AS dst_idx,
      round((10 + random() * 9990)::numeric, 2) AS tran_amount
    FROM gs
  ),
  numbered AS (
    SELECT
      p.n,
      p.tran_date,
      /* pick actual ids from arrays by index */
      v_src_ids[p.src_idx] AS src_customer_id,
      v_dst_ids[p.dst_idx] AS dst_customer_id,
      p.tran_amount,
      row_number() OVER (PARTITION BY p.tran_date ORDER BY p.n) AS seq_for_day
    FROM picked p
  )
  INSERT INTO findba.fin_transaction (
      transaction_id,
      transaction_desc,
      transaction_internal_code,
      bsn_group_id,
      status_code,
      transaction_date,
      transaction_amnt,
      src_customer_id,
      dst_customer_id,
      creation_date,
      created_by,
      cur_id,
      cou_id
  )
  SELECT
      nextval('FINDBA.S_FIN_TRANSACTION') AS transaction_id,
      'VASP' || src_customer_id::text || '-CLIENT_' || dst_customer_id::text AS transaction_desc,
      to_char(tran_date, 'YYYYMMDD') || '-' || lpad(seq_for_day::text, 2, '0') AS transaction_internal_code,
      7777                           AS bsn_group_id,
      1502                           AS status_code,
      tran_date                      AS transaction_date,
      tran_amount                    AS transaction_amnt,
      src_customer_id,
      dst_customer_id,
      current_date                   AS creation_date,
      -1995                          AS created_by,
      1                              AS cur_id,
      3535                           AS cou_id
  FROM numbered;

  RAISE NOTICE 'Inserted % transaction(s) into findba.fin_transaction', p_rows;
END;
$$;
