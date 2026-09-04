function ExecutiveOverview({
  dashboard,
  mode,
}) {

  if (!dashboard) {

    return (
      <div className="page">

        <SectionHeader
          title="Executive Overview"
          description="Retail demand forecasting dashboard."
        />

        <EmptyState
          text="Dashboard data unavailable."
        />

      </div>
    );
  }


  /*
     Prepare the daily chart data.

     Backend provides:

       date
       forecast
       actual

     In validation mode:
       forecast = model prediction
       actual   = real historical sales

     In evaluation mode:
       actual = null
  */

  const daily = safeArray(
    dashboard.daily
  ).map((row) => ({
    ...row,

    label: shortDate(row.date),

    forecast:
      Number(row.forecast ?? 0),

    actual:
      row.actual === null ||
      row.actual === undefined ||
      row.actual === ""
        ? null
        : Number(row.actual),
  }));


  return (

    <div className="page">

      <SectionHeader
        title="Executive Overview"
        description={`Demand forecast · ${
          date(dashboard.first_date)
        } → ${
          date(dashboard.last_date)
        }`}
      />


      {/* ==================================================
          KPI CARDS
      ================================================== */}

      <div className="metric-grid">

        <MetricCard
          title="Forecast Units"
          value={number(
            dashboard.total_units
          )}
          subtitle="Total predicted demand"
          icon={TrendingUp}
        />


        <MetricCard
          title="Forecast Series"
          value={number(
            dashboard.series
          )}
          subtitle="Item × Store series"
          icon={Boxes}
        />


        <MetricCard
          title="Average / Day"
          value={number(
            dashboard.avg_units_per_day
          )}
          subtitle="Across selected scope"
          icon={Activity}
        />


        <MetricCard
          title="Peak Day"
          value={number(
            dashboard.peak_day_units
          )}
          subtitle={date(
            dashboard.peak_date
          )}
          icon={Target}
        />

      </div>


      {/* ==================================================
          DAILY FORECAST / VALIDATION
      ================================================== */}

      <div className="card">

        <div className="card-header">

          <div>

            <h3>
              Daily Demand Forecast
            </h3>

            <span>
              {mode === "validation"
                ? "Actual Sales vs Forecast"
                : "Forecast window"}
            </span>

          </div>

          <span className="badge">
            28 DAYS
          </span>

        </div>


        {daily.length > 0 ? (

          <ResponsiveContainer
            width="100%"
            height={360}
          >

            <LineChart
              data={daily}
              margin={{
                top: 10,
                right: 20,
                left: 10,
                bottom: 10,
              }}
            >

              <CartesianGrid
                strokeDasharray="3 3"
                vertical={false}
              />


              <XAxis
                dataKey="label"
              />


              <YAxis />


              <Tooltip
                formatter={(value, name) => {

                  const numericValue =
                    Number(value ?? 0);

                  if (name === "Actual Sales") {

                    return [
                      numericValue.toLocaleString(
                        "en-IN",
                        {
                          maximumFractionDigits: 2,
                        }
                      ),
                      "Actual Sales",
                    ];
                  }

                  return [
                    numericValue.toLocaleString(
                      "en-IN",
                      {
                        maximumFractionDigits: 2,
                      }
                    ),
                    "Forecast",
                  ];
                }}

                labelFormatter={(label, payload) => {

                  const row =
                    payload &&
                    payload.length > 0
                      ? payload[0].payload
                      : null;

                  if (
                    row &&
                    row.date
                  ) {
                    return date(row.date);
                  }

                  return label;
                }}
              />


              <Legend />


              {/* ==========================================
                  FORECAST LINE
              ========================================== */}

              <Line
                type="monotone"
                dataKey="forecast"
                name="Forecast"
                stroke="#2563eb"
                strokeWidth={3}
                dot={false}
                activeDot={{
                  r: 6,
                }}
              />


              {/* ==========================================
                  ACTUAL SALES LINE

                  ONLY SHOWN IN VALIDATION MODE
              ========================================== */}

              {mode === "validation" && (

                <Line
                  type="monotone"
                  dataKey="actual"
                  name="Actual Sales"
                  stroke="#16a34a"
                  strokeWidth={3}
                  dot={false}
                  activeDot={{
                    r: 6,
                  }}
                  connectNulls={false}
                />

              )}

            </LineChart>

          </ResponsiveContainer>

        ) : (

          <EmptyState
            text="No daily forecast data available."
          />

        )}

      </div>


    </div>
  );
}