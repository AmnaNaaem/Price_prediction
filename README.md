# Price_prediction
# This is a tickets pricing monitoring system. It scrapes tickets pricing data periodically and stores it in a database. Ticket pricing changes based on demand and time, # and there can be significant difference in price. We are creating this product mainly with ourselves in mind. Users can set up alarms using an email, choosing an # origin and destination (cities), time (date and hour range picker) choosing a price reduction over mean price, etc.

# Following is the description for columns in the dataset

# insert_date: date and time when the price was collected and written in the database
# origin: origin city
# destination: destination city
# start_date: train departure time
# end_date: train arrival time
# train_type: train service name
# price: price
# train_class: ticket class, tourist, business, etc.
# fare: ticket fare, round trip, etc
