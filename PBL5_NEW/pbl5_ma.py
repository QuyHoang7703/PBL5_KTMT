from .extension import ma

class AccountSchema(ma.Schema):
    class Meta:
        fields = ('id_account', 'username', 'password', 'role')

class InformationSchema(ma.Schema):
    class Meta:
        fields = ('id_account', 'name', 'gender', 'phone', 'cccd', 'gmail')

class TicketSchema(ma.Schema):
    class Meta:
        fields = ('vehicle_plate', 'init_date', 'expiry', 'status', 'cccd')

class HistorySchema(ma.Schema):
    class Meta:
        fields = ('id_history', 'vehicle_plate', 'date_in', 'date_out', 'time_in', 'time_out')
